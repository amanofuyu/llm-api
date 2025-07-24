import { log } from 'node:console'
import { GoogleGenAI } from '@google/genai'
import { Hono } from 'hono'
import { cors } from 'hono/cors'
import { validator } from 'hono/validator'
import OpenAI from 'openai'
import { z } from 'zod'

interface Bindings {
  API_KEY: string
  GEMINI_KEY: string
}

const app = new Hono<{ Bindings: Bindings }>()

app.use('*', cors())

app.get('/', (c) => {
  return c.text('Hello World')
})

interface LLM {
  model: string
  from: string
  input: Array<'text' | 'image' | 'audio' | 'video' | 'file' | 'structured'>
  output: Array<'text' | 'image' | 'structured'>
}

app.get('/model/list', (c) => {
  return c.json({
    code: 200,
    data: [
      {
        model: 'THUDM/GLM-4.1V-9B-Thinking',
        from: 'siliconflow',
        input: ['text'],
        output: ['text'],
      },
      {
        model: 'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',
        from: 'siliconflow',
        input: ['text'],
        output: ['text'],
      },
      {
        model: 'Qwen/Qwen3-8B',
        from: 'siliconflow',
        input: ['text'],
        output: ['text'],
      },
      {
        model: 'FunAudioLLM/SenseVoiceSmall',
        from: 'siliconflow',
        input: ['audio'],
        output: ['text'],
      },
      {
        model: 'Kwai-Kolors/Kolors',
        from: 'siliconflow',
        input: ['text'],
        output: ['image'],
      },
    ] as LLM[],
    message: 'success',
    type: 'success',
  })
})

// 定义验证Schema
const ChatMessageSchema = z.object({
  role: z.enum(['user', 'assistant', 'system']),
  content: z.string().min(1, '消息内容不能为空'),
})

const ChatRequestSchema = z.object({
  model: z.string().optional().default('Qwen/Qwen2.5-7B-Instruct'),
  messages: z.array(ChatMessageSchema).min(1, '至少需要一条消息'),
  stream: z.boolean().optional().default(true),
  temperature: z.number().min(0).max(2).optional(),
  max_tokens: z.number().min(1).max(4000).optional(),
  top_p: z.number().min(0).max(1).optional(),
})

app.post(
  '/chat',
  validator('json', (value, c) => {
    const parsed = ChatRequestSchema.safeParse(value)
    if (!parsed.success) {
      return c.json({ error: parsed.error.message }, 400)
    }
    return parsed.data
  }),
  async (c) => {
    try {
      const client = new OpenAI({
        apiKey: c.env.API_KEY,
        baseURL: 'https://api.siliconflow.cn/v1',
      })

      const body = c.req.valid('json')

      if (body.stream) {
        const response = await client.chat.completions.create({
          model: body.model,
          messages: body.messages,
          stream: body.stream,
          temperature: body.temperature,
          max_tokens: body.max_tokens,
          top_p: body.top_p,
        })

        const stream = new ReadableStream({
          async start(controller) {
            try {
              for await (const chunk of response) {
                const chunkText = `data: ${JSON.stringify(chunk)}\n\n`
                controller.enqueue(new TextEncoder().encode(chunkText))
              }
              controller.enqueue(new TextEncoder().encode('data: [DONE]\n\n'))
              controller.close()
            }
            catch (error) {
              controller.error(error)
            }
          },
        })

        return new Response(stream, {
          headers: {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
          },
        })
      }
      else {
        const response = await client.chat.completions.create({
          model: body.model,
          messages: body.messages,
          stream: body.stream,
          temperature: body.temperature,
          max_tokens: body.max_tokens,
          top_p: body.top_p,
        })

        return c.json({
          code: 200,
          data: response,
          message: 'success',
          redirect_url: '',
          toast: 0,
          type: 'success',
        })
      }
    }
    catch (error) {
      log('Error:', error)
      return c.json({ error: '服务器内部错误' }, 500)
    }
  },
)

app.post('/transcriptions', async (c) => {
  const formData = await c.req.formData()

  formData.append('model', 'FunAudioLLM/SenseVoiceSmall')

  const options = {
    method: 'POST',
    headers: { Authorization: `Bearer ${c.env.API_KEY}` },
    body: formData,
  }

  return fetch('https://api.siliconflow.cn/v1/audio/transcriptions', options)
})

app.get('/gemini', async (c) => {
  const ai = new GoogleGenAI({
    apiKey: c.env.GEMINI_KEY,
  })

  const response = await ai.models.generateContent({
    model: 'gemini-2.5-flash',
    contents: 'Explain how AI works in a few words',
    // config: {
    //   thinkingConfig: {
    //     thinkingBudget: 0, // Disables thinking
    //   },
    // }
  })

  return c.json(response)
})

export default app
