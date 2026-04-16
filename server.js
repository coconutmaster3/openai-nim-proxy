// server.js - OpenAI to NVIDIA NIM API Proxy (Fixed for Janitor AI & Chub AI)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors({
  origin: '*',
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With']
}));
app.use(express.json({ limit: '10mb' }));
app.options('*', cors());

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

if (!NIM_API_KEY) {
  console.error('⚠️  NIM_API_KEY environment variable is not set!');
}

const SHOW_REASONING = process.env.SHOW_REASONING === 'true';
const ENABLE_THINKING_MODE = process.env.ENABLE_THINKING_MODE === 'true';

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

// Request logging middleware
app.use('/v1/chat/completions', (req, res, next) => {
  console.log(`─── Incoming request ───`);
  console.log(`Model: ${req.body?.model || 'NONE'}`);
  console.log(`Stream: ${req.body?.stream}`);
  console.log(`Messages: ${req.body?.messages?.length || 0} messages`);
  // Log any extra params Chub is sending
  const extras = Object.keys(req.body || {}).filter(k =>
    !['model', 'messages', 'stream', 'temperature', 'top_p', 'max_tokens', 'stop', 'presence_penalty', 'frequency_penalty'].includes(k)
  );
  if (extras.length > 0) {
    console.log(`Stripping unsupported params: ${extras.join(', ')}`);
  }
  next();
});

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

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

// Params that NVIDIA NIM actually supports
const ALLOWED_PARAMS = [
  'model', 'messages', 'stream', 'temperature', 'top_p',
  'max_tokens', 'max_completion_tokens', 'stop', 'seed',
  'frequency_penalty', 'presence_penalty', 'repetition_penalty',
  'reasoning_effort'
];

app.post('/v1/chat/completions', async (req, res) => {
  const startTime = Date.now();

  try {
    const { model, messages, stream = false } = req.body;

    if (!model) {
      console.error('❌ No model provided');
      return res.status(400).json({ error: { message: 'model is required', type: 'invalid_request_error' } });
    }

    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      console.error('❌ No messages provided');
      return res.status(400).json({ error: { message: 'messages is required', type: 'invalid_request_error' } });
    }

    // Build clean body — ONLY include params NIM supports
    let nimModel = model;
    if (model === 'kimi-k2-thinking') {
      nimModel = 'moonshotai/kimi-k2-thinking';
    }

    const nimBody = { model: nimModel, messages, stream };

    // Copy only allowed params
    for (const key of ALLOWED_PARAMS) {
      if (key === 'model' || key === 'messages' || key === 'stream') continue;
      if (req.body[key] !== undefined) {
        nimBody[key] = req.body[key];
      }
    }

    // Normalize max_tokens (Chub sometimes sends max_completion_tokens)
    if (nimBody.max_completion_tokens && !nimBody.max_tokens) {
      nimBody.max_tokens = nimBody.max_completion_tokens;
      delete nimBody.max_completion_tokens;
    }

    // Thinking mode
    if (ENABLE_THINKING_MODE && (model.includes('thinking') || model.includes('deepseek-r1'))) {
      nimBody.reasoning_effort = 'high';
    }

    const headers = {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${NIM_API_KEY}`,
    };

    if (stream) {
      // ─── Streaming ───
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

        console.log(`✅ Stream connected to NIM in ${Date.now() - startTime}ms`);

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
          console.log(`✅ Stream completed in ${Date.now() - startTime}ms`);
          res.write('data: [DONE]\n\n');
          res.end();
        });

        response.data.on('error', (err) => {
          console.error(`❌ Stream error after ${Date.now() - startTime}ms:`, err.message);
          if (!res.headersSent) {
            res.status(502).json({ error: { message: 'Upstream stream error' } });
          } else {
            res.end();
          }
        });

      } catch (err) {
        const elapsed = Date.now() - startTime;
        console.error(`❌ Stream setup failed after ${elapsed}ms:`, err.code, err.message);
        if (err.response) {
          console.error(`   NIM returned ${err.response.status}:`, JSON.stringify(err.response.data).slice(0, 200));
        }
        if (!res.headersSent) {
          res.status(502).json({ error: { message: `Failed to connect to NIM: ${err.message}` } });
        } else {
          res.end();
        }
      }

    } else {
      // ─── Non-streaming ───
      const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimBody, {
        headers,
        timeout: 300000,
      });

      console.log(`✅ Non-stream completed in ${Date.now() - startTime}ms`);

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
    const elapsed = Date.now() - startTime;
    console.error(`❌ Request failed after ${elapsed}ms:`, err.code || 'UNKNOWN', err.message);
    if (err.response) {
      console.error(`   NIM returned ${err.response.status}:`, JSON.stringify(err.response.data).slice(0, 300));
      res.status(err.response.status).json(err.response.data);
    } else {
      res.status(500).json({ error: { message: err.message, type: 'proxy_error' } });
    }
  }
});

app.use('*', (req, res) => {
  res.status(404).json({ error: { message: 'Not found', type: 'not_found' } });
});

// Start with retry
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
        console.error(`❌ Port ${PORT} still in use after all retries.`);
        process.exit(1);
      }
    } else {
      console.error('Server error:', err);
      process.exit(1);
    }
  });

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
