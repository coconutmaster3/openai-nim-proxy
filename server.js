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

// Handle preflight
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

// Root endpoint - some platforms check this
app.get('/', (req, res) => {
  res.json({ 
    status: 'ok', 
    message: 'OpenAI-compatible NVIDIA NIM Proxy',
    endpoints: ['/v1/models', '/v1/chat/completions', '/health']
  });
});

// List models endpoint - OpenAI compatible format
app.get('/v1/models', (req, res) => {
  const models = MODELS.map(model => ({
    id: model,
    object: 'model',
    created: Math.floor(Date.now() / 1000),
    owned_by: 'nvidia-nim',
    permission: [],
    root: model,
    parent: null
  }));
  
  res.json({
    object: 'list',
    data: models
  });
});

// Also support /models without /v1 prefix (some platforms use this)
app.get('/models', (req, res) => {
  res.redirect('/v1/models');
});

// Chat completions endpoint
app.post('/v1/chat/completions', async (req, res) => {
  const requestId = `chatcmpl-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  
  try {
    const { model, messages, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, stop, stream } = req.body;
    
    // Validate model
    if (!model) {
      return res.status(400).json({
        error: {
          message: 'Model is required',
          type: 'invalid_request_error',
          param: null,
          code: 'model_missing'
        }
      });
    }
    
    // Validate messages
    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({
        error: {
          message: 'Messages array is required and must not be empty',
          type: 'invalid_request_error',
          param: null,
          code: 'messages_required'
        }
      });
    }
    
    console.log(`[${new Date().toISOString()}] Request: model=${model}, messages=${messages.length}, stream=${!!stream}`);
    
    // Build NIM request - ONLY include parameters that are set
    const nimRequest = {
      model: model,
      messages: messages,
      max_tokens: max_tokens || 4096,
      stream: !!stream
    };
    
    // Only add optional params if they're defined
    if (temperature !== undefined && temperature !== null) {
      nimRequest.temperature = temperature;
    }
    if (top_p !== undefined && top_p !== null) {
      nimRequest.top_p = top_p;
    }
    if (frequency_penalty !== undefined && frequency_penalty !== null) {
      nimRequest.frequency_penalty = frequency_penalty;
    }
    if (presence_penalty !== undefined && presence_penalty !== null) {
      nimRequest.presence_penalty = presence_penalty;
    }
    if (stop !== undefined && stop !== null) {
      nimRequest.stop = stop;
    }
    
    // Add thinking mode if enabled
    if (ENABLE_THINKING_MODE) {
      nimRequest.chat_template_kwargs = { thinking: true };
    }
    
    // Stream options
    if (stream) {
      nimRequest.stream_options = { include_usage: true };
    }
    
    // Make request to NVIDIA NIM API
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json',
        'Accept': stream ? 'text/event-stream' : 'application/json'
      },
      responseType: stream ? 'stream' : 'json',
      timeout: 180000,
      validateStatus: (status) => status < 500 // Don't throw on 4xx errors
    });
    
    // Handle non-success status codes from NIM
    if (response.status !== 200) {
      console.error(`NIM API error: ${response.status}`, response.data);
      return res.status(response.status).json({
        error: {
          message: response.data?.error?.message || response.data?.detail || `NIM API error: ${response.status}`,
          type: 'upstream_error',
          param: null,
          code: response.status
        }
      });
    }
    
    if (stream) {
      // Handle streaming response
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache, no-transform');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-Accel-Buffering', 'no');
      res.flushHeaders();
      
      let buffer = '';
      let reasoningStarted = false;
      let sentAnyContent = false;
      
      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        for (const line of lines) {
          const trimmedLine = line.trim();
          
          if (!trimmedLine || trimmedLine === '') continue;
          
          if (trimmedLine === 'data: [DONE]') {
            res.write('data: [DONE]\n\n');
            continue;
          }
          
          if (!trimmedLine.startsWith('data: ')) continue;
          
          try {
            const data = JSON.parse(trimmedLine.slice(6));
            
            // Ensure required fields exist
            if (!data.id) data.id = requestId;
            if (!data.object) data.object = 'chat.completion.chunk';
            if (!data.created) data.created = Math.floor(Date.now() / 1000);
            if (!data.model) data.model = model;
            
            if (data.choices && data.choices[0]) {
              const delta = data.choices[0].delta || {};
              const reasoning = delta.reasoning_content;
              const content = delta.content;
              
              // Handle reasoning display
              if (SHOW_REASONING && reasoning) {
                let combinedContent = '';
                
                if (!reasoningStarted) {
                  combinedContent = '<think>\n' + reasoning;
                  reasoningStarted = true;
                } else {
                  combinedContent = reasoning;
                }
                
                delta.content = combinedContent;
                delete delta.reasoning_content;
                sentAnyContent = true;
              } else if (SHOW_REASONING && content && reasoningStarted) {
                delta.content = '\n</think>\n\n' + content;
                reasoningStarted = false;
                sentAnyContent = true;
              } else if (content) {
                // Normal content, no reasoning
                delta.content = content;
                sentAnyContent = true;
              } else if (reasoning) {
                // Reasoning exists but SHOW_REASONING is false - hide it
                delete delta.reasoning_content;
                // Don't send empty chunks - skip entirely
                continue;
              } else {
                // No content at all (e.g., role-only chunk) - send as-is
                if (!delta.role && !delta.content) {
                  continue; // Skip truly empty chunks
                }
              }
              
              // Ensure delta has at least role on first chunk
              if (!sentAnyContent && !delta.role) {
                delta.role = 'assistant';
              }
            }
            
            res.write(`data: ${JSON.stringify(data)}\n\n`);
            
          } catch (e) {
            // Skip malformed JSON chunks silently
          }
        }
      });
      
      response.data.on('end', () => {
        res.write('data: [DONE]\n\n');
        res.end();
      });
      
      response.data.on('error', (err) => {
        console.error('Stream error:', err.message);
        if (!res.headersSent) {
          res.status(500).json({
            error: {
              message: 'Stream error from upstream',
              type: 'upstream_error',
              code: 500
            }
          });
        } else {
          res.write('data: [DONE]\n\n');
          res.end();
        }
      });
      
      req.on('close', () => {
        response.data.destroy();
      });
      
    } else {
      // Non-streaming response - ensure exact OpenAI format
      const nimData = response.data;
      const choices = (nimData.choices || []).map((choice, index) => {
        let fullContent = choice.message?.content || '';
        
        // Handle reasoning
        if (SHOW_REASONING && choice.message?.reasoning_content) {
          fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
        }
        
        return {
          index: choice.index !== undefined ? choice.index : index,
          message: {
            role: choice.message?.role || 'assistant',
            content: fullContent
          },
          finish_reason: choice.finish_reason || 'stop'
        };
      });
      
      const openaiResponse = {
        id: nimData.id || requestId,
        object: 'chat.completion',
        created: nimData.created || Math.floor(Date.now() / 1000),
        model: nimData.model || model,
        system_fingerprint: nimData.system_fingerprint || `fp_${Date.now()}`,
        choices: choices,
        usage: nimData.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };
      
      res.json(openaiResponse);
    }
    
  } catch (error) {
    console.error('Proxy error:', error.message);
    
    const status = error.response?.status || 500;
    const message = error.code === 'ECONNABORTED' 
      ? 'Request to NVIDIA NIM timed out'
      : error.response?.data?.error?.message || error.response?.data?.detail || error.message || 'Internal server error';
    
    res.status(status).json({
      error: {
        message: message,
        type: status === 401 ? 'authentication_error' : 
              status === 429 ? 'rate_limit_error' : 
              status === 404 ? 'not_found_error' : 'api_error',
        param: null,
        code: status
      }
    });
  }
});

// Also support /chat/completions without /v1 prefix
app.post('/chat/completions', (req, res) => {
  req.url = '/v1/chat/completions';
  app.handle(req, res);
});

// Catch-all 404
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Unknown endpoint: ${req.method} ${req.path}`,
      type: 'invalid_request_error',
      param: null,
      code: 404
    }
  });
});

app.listen(PORT, '0.0.0.0', () => {
  console.log('='.repeat(50));
  console.log('OpenAI to NVIDIA NIM Proxy');
  console.log('='.repeat(50));
  console.log(`Port: ${PORT}`);
  console.log(`NIM Base: ${NIM_API_BASE}`);
  console.log(`API Key: ${NIM_API_KEY ? '***' + NIM_API_KEY.slice(-4) : 'NOT SET!'}`);
  console.log(`Show Reasoning: ${SHOW_REASONING}`);
  console.log(`Thinking Mode: ${ENABLE_THINKING_MODE}`);
  console.log(`Models: ${MODELS.length}`);
  console.log('='.repeat(50));
});
// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'OpenAI to NVIDIA NIM Proxy', 
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    config_source: {
      show_reasoning: process.env.SHOW_REASONING ? 'environment' : 'default',
      enable_thinking: process.env.ENABLE_THINKING_MODE ? 'environment' : 'default'
    }
  });
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = MODELS.map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  
  res.json({
    object: 'list',
    data: models
  });
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    
    // Pass model directly to NIM (no mapping)
    const nimModel = model;
    
    // Transform OpenAI request to NIM format
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 9024,
      extra_body: ENABLE_THINKING_MODE 
        ? { chat_template_kwargs: { thinking: true } } 
        : undefined,
      stream: stream || false,
      stream_options: stream ? { include_usage: true } : undefined
    };
    
    // Make request to NVIDIA NIM API
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json',
      timeout: 120000
    });
    
    if (stream) {
      // Handle streaming response with reasoning
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.flushHeaders();
      
      let buffer = '';
      let reasoningStarted = false;
      
      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              res.write(line + '\n\n');
              return;
            }
            
            try {
              const data = JSON.parse(line.slice(6));
              if (data.choices?.[0]?.delta) {
                const reasoning = data.choices[0].delta.reasoning_content;
                const content = data.choices[0].delta.content;
                
                if (SHOW_REASONING) {
                  let combinedContent = '';
                  
                  if (reasoning && !reasoningStarted) {
                    combinedContent = '>{!!}\n' + reasoning;
                    reasoningStarted = true;
                  } else if (reasoning) {
                    combinedContent = reasoning;
                  }
                  
                  if (content && reasoningStarted) {
                    combinedContent += '!!<\n\n' + content;
                    reasoningStarted = false;
                  } else if (content) {
                    combinedContent += content;
                  }
                  
                  if (combinedContent) {
                    data.choices[0].delta.content = combinedContent;
                    delete data.choices[0].delta.reasoning_content;
                  }
                } else {
                  if (content) {
                    data.choices[0].delta.content = content;
                  } else {
                    data.choices[0].delta.content = '';
                  }
                  delete data.choices[0].delta.reasoning_content;
                }
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              console.error('Parse error (skipping):', e.message);
            }
          }
        });
      });
      
      response.data.on('end', () => {
        res.write('data: [DONE]\n\n');
        res.end();
      });
      
      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        res.write('data: [DONE]\n\n');
        res.end();
      });
      
      req.on('close', () => {
        console.log('Client disconnected');
        response.data.destroy();
      });
    } else {
      // Transform NIM response to OpenAI format with reasoning
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';
          
          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = '>{!!}\n' + choice.message.reasoning_content + '\n!!<\n\n' + fullContent;
          }
          
          return {
            index: choice.index,
            message: {
              role: choice.message.role,
              content: fullContent
            },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };
      
      res.json(openaiResponse);
    }
    
  } catch (error) {
    console.error('Proxy error:', error.message);
    
    res.status(error.response?.status || 500).json({
      error: {
        message: error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: error.response?.status || 500
      }
    });
  }
});

// Catch-all for unsupported endpoints
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
});
