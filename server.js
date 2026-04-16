// server.js - OpenAI to NVIDIA NIM API Proxy (Fixed for Janitor AI & Chub AI)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors({
  origin: '*',
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With']
}));
app.use(express.json({ limit: '10mb' }));
app.options('*', cors());

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

if (!NIM_API_KEY) {
  console.error('⚠️  NIM_API_KEY environment variable is not set!');
}

// Toggles
const SHOW_REASONING = process.env.SHOW_REASONING === 'true';
const ENABLE_THINKING_MODE = process.env.ENABLE_THINKING_MODE === 'true';

// Available models
const MODELS = [
  'nemotron-3-super-120b-a12b',
  'gemma-4-31b-it',
  'kimi-k2.5',
  'deepseek-v3.2',
  'moonshotai/kimi-k2-thinking',
  'qwen3-next-80b-a3b-instruct',
  'qwen3.5-397b-a17b',
  'llama-3.3-70b-instruct',
  'llama-3.1-8b-instruct'
];

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

// OpenAI-compatible /v1/models
app.get('/v1/models', (req, res) => {
  res.json({
    object: 'list',
    data: MODELS.map(id => ({
      id,
      object: 'model',
      created: Math.floor(Date.now() / 1000),
      owned_by: 'nvidia',
      permission: [],
      root: id,
      parent: null
    }))
  });
});

// OpenAI-compatible /v1/chat/completions
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, stream = false, temperature, top_p, max_tokens, stop, presence_penalty, frequency_penalty } = req.body;

    if (!model) {
      return res.status(400).json({ error: { message: 'model is required', type: 'invalid_request_error' } });
    }

    let nimModel = model;
    if (model === 'kimi-k2-thinking') {
      nimModel = 'moonshotai/kimi-k2-thinking';
    }

    const nimBody = { model: nimModel, messages, stream };
    if (temperature !== undefined) nimBody.temperature = temperature;
    if (top_p !== undefined) nimBody.top_p = top_p;
    if (max_tokens !== undefined) nimBody.max_tokens = max_tokens;
    if (stop !== undefined) nimBody.stop = stop;
    if (presence_penalty !== undefined) nimBody.presence_penalty = presence_penalty;
    if (frequency_penalty !== undefined) nimBody.frequency_penalty = frequency_penalty;

    if (ENABLE_THINKING_MODE && (model.includes('thinking') || model.includes('deepseek-r1'))) {
      nimBody.reasoning_effort = 'high';
    }

    const headers = {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${NIM_API_KEY}`,
    };

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-Accel-Buffering', 'no');

      try {
        const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimBody, {
          headers,
          responseType: 'stream',
          timeout: 300000,
        });

        response.data.on('data', (chunk) => {
          const raw = chunk.toString();
          if (!SHOW_REASONING) {
            const lines = raw.split('\n').filter(line => {
              if (!line.startsWith('data: ')) return true;
              const data = line.slice(6).trim();
              if (data === '[DONE]') return true;
              try {
                const parsed = JSON.parse(data);
                if (parsed.choices?.[0]?.delta?.reasoning_content) return false;
                if (parsed.choices?.[0]?.delta?.reasoning) return false;
              } catch (e) {}
              return true;
            });
            res.write(lines.join('\n') + '\n');
          } else {
            res.write(raw);
          }
        });

        response.data.on('end', () => {
          res.write('data: [DONE]\n\n');
          res.end();
        });

        response.data.on('error', (err) => {
          console.error('Stream error from NIM:', err.message);
          if (!res.headersSent) {
            res.status(502).json({ error: { message: 'Upstream stream error' } });
          } else {
            res.end();
          }
        });

      } catch (err) {
        console.error('Stream setup error:', err.message);
        if (!res.headersSent) {
          res.status(502).json({ error: { message: 'Failed to connect to NIM' } });
        } else {
          res.end();
        }
      }

    } else {
      const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimBody, {
        headers,
        timeout: 300000,
      });

      let data = response.data;

      if (!SHOW_REASONING && data.choices) {
        data.choices = data.choices.map(choice => {
          if (choice.message?.reasoning_content) delete choice.message.reasoning_content;
          if (choice.message?.reasoning) delete choice.message.reasoning;
          return choice;
        });
        if (data.usage?.prompt_tokens_details?.reasoning_tokens !== undefined) {
          delete data.usage.prompt_tokens_details.reasoning_tokens;
        }
      }

      res.json(data);
    }

  } catch (err) {
    console.error('Completions error:', err.message);
    if (err.response) {
      res.status(err.response.status).json(err.response.data);
    } else {
      res.status(500).json({ error: { message: err.message, type: 'proxy_error' } });
    }
  }
});

// Catch-all 404
app.use('*', (req, res) => {
  res.status(404).json({ error: { message: 'Not found', type: 'not_found' } });
});

// ─── Start server with retry logic for EADDRINUSE ───
function startServer(retries = 5, delay = 3000) {
  const server = app.listen(PORT, '0.0.0.0', () => {
    console.log(`✅ OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
    console.log(`   Health check: http://localhost:${PORT}/health`);
    console.log(`   Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
    console.log(`   Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
  });

  server.on('error', (err) => {
    if (err.code === 'EADDRINUSE') {
      if (retries > 0) {
        console.warn(`⚠️  Port ${PORT} in use, retrying in ${delay / 1000}s... (${retries} left)`);
        server.close();
        setTimeout(() => startServer(retries - 1, delay), delay);
      } else {
        console.error(`❌ Port ${PORT} still in use after all retries. Giving up.`);
        process.exit(1);
      }
    } else {
      console.error('Server error:', err);
      process.exit(1);
    }
  });

  // Graceful shutdown
  process.on('SIGTERM', () => {
    console.log('SIGTERM received, shutting down...');
    server.close(() => process.exit(0));
  });

  process.on('SIGINT', () => {
    console.log('SIGINT received, shutting down...');
    server.close(() => process.exit(0));
  });
}

startServer();
