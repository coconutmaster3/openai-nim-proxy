const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors({
  origin: '*',
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS', 'HEAD'],
  allowedHeaders: '*',
  exposedHeaders: '*',
  maxAge: 86400,
  credentials: false
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

// Use your full prefixed model IDs that Janitor is already happy with
const MODELS = [
  'meta/llama-3.3-70b-instruct',
  'meta/llama-3.1-8b-instruct',
  'moonshotai/kimi-k2-thinking',
  'moonshotai/kimi-k2-instruct',
  'deepseek-ai/deepseek-v3.2',
  'deepseek-ai/deepseek-r1',
  'nvidia/llama-3.1-nemotron-70b-instruct',
  'google/gemma-4-31b-it',
  'qwen/qwen3-next-80b-a3b-instruct',
  'qwen/qwen3.5-397b-a17b',
];

function extractErrorMessage(err) {
  if (err.response?.data && typeof err.response.data.pipe === 'function') {
    return `${err.response.status} ${err.response.statusText} from ${err.config?.url || 'upstream'}`;
  }
  if (err.response?.data) {
    const d = err.response.data;
    if (typeof d === 'string') return d.trim().slice(0, 500);
    if (typeof d === 'object' && d !== null) {
      const msg = d.message || d.error?.message || d.detail || d.msg
        || (d.error && typeof d.error === 'string' ? d.error : null);
      if (msg) return String(msg).slice(0, 500);
      return `${err.response.status} ${err.response.statusText}`;
    }
    return String(d).slice(0, 500);
  }
  if (err.code) return `${err.code}: ${err.message || 'unknown'}`;
  return (err.message || 'Unknown error').slice(0, 500);
}

app.use('/v1/chat/completions', (req, res, next) => {
  console.log(`─── Incoming request ───`);
  console.log(`Model: ${req.body?.model || 'NONE'}`);
  console.log(`Stream: ${req.body?.stream}`);
  console.log(`Messages: ${req.body?.messages?.length || 0} messages`);
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
      id, object: 'model', created: Math.floor(Date.now() / 1000),
      owned_by: 'nvidia', permission: [], root: id, parent: null
    }))
  });
});

const ALLOWED_PARAMS = [
  'model', 'messages', 'stream', 'temperature', 'top_p',
  'max_tokens', 'max_completion_tokens', 'stop', 'seed',
  'frequency_penalty', 'presence_penalty', 'repetition_penalty',
  'reasoning_effort'
];

function sendError(res, status, message, type = 'proxy_error') {
  if (!res.headersSent) {
    res.status(status).json({ error: { message, type, code: status } });
  } else {
    res.end();
  }
}

app.post('/v1/chat/completions', async (req, res) => {
  const startTime = Date.now();

  try {
    const { model, messages, stream = false } = req.body;

    if (!model) return sendError(res, 400, 'model is required', 'invalid_request_error');
    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      return sendError(res, 400, 'messages is required', 'invalid_request_error');
    }

    const nimBody = { model, messages, stream };
    for (const key of ALLOWED_PARAMS) {
      if (key === 'model' || key === 'messages' || key === 'stream') continue;
      if (req.body[key] !== undefined) nimBody[key] = req.body[key];
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
          headers, responseType: 'stream', timeout: 300000,
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
        const msg = extractErrorMessage(err);
        console.error(`❌ Stream setup failed after ${elapsed}ms [${status}]: ${msg}`);

        if (status === 404) sendError(res, 404, `Model '${model}' not found on NVIDIA NIM.`, 'model_not_found');
        else if (status === 401) sendError(res, 401, 'Invalid NIM API key', 'authentication_error');
        else if (status === 429) sendError(res, 429, 'Rate limit exceeded.', 'rate_limit_error');
        else sendError(res, status, `NIM error: ${msg}`, 'upstream_error');
      }

    } else {
      const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimBody, {
        headers, timeout: 300000,
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
    const msg = extractErrorMessage(err);
    console.error(`❌ Request failed after ${elapsed}ms [${status}]: ${msg}`);

    if (status === 404) sendError(res, 404, `Model '${req.body?.model}' not found on NVIDIA NIM.`, 'model_not_found');
    else if (status === 401) sendError(res, 401, 'Invalid NIM API key', 'authentication_error');
    else if (status === 429) sendError(res, 429, 'Rate limit exceeded.', 'rate_limit_error');
    else sendError(res, status, `Proxy error: ${msg}`, 'proxy_error');
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
      console.error('Server error:', err.message);
      process.exit(1);
    }
  });

  process.on('SIGTERM', () => { console.log('SIGTERM received, shutting down...'); server.close(() => process.exit(0)); });
  process.on('SIGINT', () => { console.log('SIGINT received, shutting down...'); server.close(() => process.exit(0)); });
}

startServer();
