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

// Safe JSON stringify — avoids circular reference crashes from axios errors
function safeStringify(obj, depth = 2) {
  const seen = new WeakSet();
  try {
    return JSON.stringify(obj, (key, value) => {
      if (typeof value === 'object' && value !== null) {
        if (seen.has(value)) return '[Circular]';
        seen.add(value);
      }
      return value;
    }, depth);
  } catch (e) {
    return String(obj);
  }
}

// Request logging
app.use('/v1/chat/completions', (req, res, next) => {
  console.log(`─── Incoming request ───`);
  console.log(`Model: ${req.body?.model || 'NONE'}`);
  console.log(`Stream: ${req.body?.stream}`);
  console.log(`Messages: ${req.body?.messages?.length || 0} messages`);
  const extras = Object.keys(req.body || {}).filter(k =>
    !['model', 'messages', 'stream', 'temperature', 'top_p', 'max_tokens', 'max_completion_tokens', 'stop', 'presence_penalty', 'frequency_penalty', 'seed', 'repetition_penalty'].includes(k)
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

const ALLOWED_PARAMS = [
  'model', 'messages', 'stream', 'temperature', 'top_p',
  'max_tokens', 'max_completion_tokens', 'stop', 'seed',
  'frequency_penalty', 'presence_penalty', 'repetition_penalty',
  'reasoning_effort'
];

// Format an error response in OpenAI format so Chub/Janitor can display it
function sendError(res, status, message, type = 'proxy_error') {
  if (!res.headersSent) {
    res.status(status).json({
      error: {
        message,
        type,
        code: status
      }
    });
  } else {
    res.end();
  }
}

app.post('/v1/chat/completions', async (req, res) => {
  const startTime = Date.now();

  try {
    const { model, messages, stream = false } = req.body;

    if (!model) {
      return sendError(res, 400, 'model is required', 'invalid_request_error');
    }

    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      return sendError(res, 400, 'messages is required', 'invalid_request_error');
    }

    // Map model name
    let nimModel = model;
    if (model === 'kimi-k2-thinking') {
      nimModel = 'moonshotai/kimi-k2-thinking';
    }

    // Build clean body
    const nimBody = { model: nimModel, messages, stream };

    for (const key of ALLOWED_PARAMS) {
      if (key === 'model' || key === 'messages' || key === 'stream') continue;
      if (req.body[key] !== undefined) {
        nimBody[key] = req.body[key];
      }
    }

    if (nimBody.max_completion_tokens && !nimBody.max_tokens) {
      nimBody.max_tokens = nimBody.max_completion_tokens;
      delete nimBody.max_completion_tokens;
    }

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

        console.log(`✅ Stream connected in ${Date.now() - startTime}ms`);

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
          console.log(`✅ Stream done in ${Date.now() - startTime}ms`);
          res.write('data: [DONE]\n\n');
          res.end();
        });

        response.data.on('error', (err) => {
          console.error(`❌ Stream read error after ${Date.now() - startTime}ms:`, err.message);
          sendError(res, 502, `Upstream stream error: ${err.message}`);
        });

      } catch (err) {
        const elapsed = Date.now() - startTime;
        const status = err.response?.status || 502;
        const msg = err.response?.data
          ? safeStringify(err.response.data)
          : err.message;

        console.error(`❌ Stream setup failed after ${elapsed}ms [${status}]:`, msg);

        // 404 = model not found on NIM
        if (status === 404) {
          sendError(res, 404, `Model '${model}' not found on NVIDIA NIM. Available models: ${MODELS.join(', ')}`, 'model_not_found');
        } else if (status === 401) {
          sendError(res, 401, 'Invalid NIM API key', 'authentication_error');
        } else {
          sendError(res, status, `NIM error: ${msg}`, 'upstream_error');
        }
      }

    } else {
      // ─── Non-streaming ───
      const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimBody, {
        headers,
        timeout: 300000,
      });

      console.log(`✅ Non-stream done in ${Date.now() - startTime}ms`);

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
    const status = err.response?.status || 500;
    const msg = err.response?.data
      ? safeStringify(err.response.data)
      : err.message;

    console.error(`❌ Request failed after ${elapsed}ms [${status}]:`, msg);

    if (status === 404) {
      sendError(res, 404, `Model '${req.body?.model}' not found on NVIDIA NIM. Available: ${MODELS.join(', ')}`, 'model_not_found');
    } else if (status === 401) {
      sendError(res, 401, 'Invalid NIM API key', 'authentication_error');
    } else {
      sendError(res, status, `Proxy error: ${msg}`, 'proxy_error');
    }
  }
});

app.use('*', (req, res) => {
  res.status(404).json({ error: { message: 'Not found', type: 'not_found' } });
});

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
