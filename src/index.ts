import { Hono } from 'hono'
import { log } from 'console'
import OpenAI from 'openai';
import { cors } from 'hono/cors';
import { z } from 'zod'
import { validator } from 'hono/validator';

type Bindings = {
  API_KEY: string
}

const app = new Hono<{ Bindings: Bindings }>()

app.use('*', cors())

app.get('/', (c) => {
  log(c.req.header())
  return c.json(c)
})

app.get('/model/list', (c) => {
  return c.json({
    code: 200,
    data: [
      {
        model: 'THUDM/GLM-4.1V-9B-Thinking',
        description: 'GLM-4.1V-9B-Thinking 是由智谱 AI 和清华大学 KEG 实验室联合发布的一款开源视觉语言模型（VLM），专为处理复杂的多模态认知任务而设计。该模型基于 GLM-4-9B-0414 基础模型，通过引入“思维链”（Chain-of-Thought）推理机制和采用强化学习策略，显著提升了其跨模态的推理能力和稳定性。作为一个 9B 参数规模的轻量级模型，它在部署效率和性能之间取得了平衡，在 28 项权威评测基准中，有 18 项的表现持平甚至超越了 72B 参数规模的 Qwen-2.5-VL-72B。该模型不仅在图文理解、数学科学推理、视频理解等任务上表现卓越，还支持高达 4K 分辨率的图像和任意宽高比输入',
      },
      {
        model: 'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',
        description: 'DeepSeek-R1-0528-Qwen3-8B 是通过从 DeepSeek-R1-0528 模型蒸馏思维链到 Qwen3 8B Base 获得的模型。该模型在开源模型中达到了最先进（SOTA）的性能，在 AIME 2024 测试中超越了 Qwen3 8B 10%，并达到了 Qwen3-235B-thinking 的性能水平。该模型在数学推理、编程和通用逻辑等多个基准测试中表现出色，其架构与 Qwen3-8B 相同，但共享 DeepSeek-R1-0528 的分词器配置'
      },
      {
        model: 'Qwen/Qwen3-8B',
        description: 'Qwen3-8B 是通义千问系列的最新大语言模型，拥有 8.2B 参数量。该模型独特地支持在思考模式（适用于复杂逻辑推理、数学和编程）和非思考模式（适用于高效的通用对话）之间无缝切换，显著增强了推理能力。模型在数学、代码生成和常识逻辑推理上表现优异，并在创意写作、角色扮演和多轮对话等方面展现出卓越的人类偏好对齐能力。此外，该模型支持 100 多种语言和方言，具备出色的多语言指令遵循和翻译能力'
      },
      {
        model: 'FunAudioLLM/SenseVoiceSmall',
        description: 'SenseVoice 是一个具有多种语音理解能力的语音基础模型，包括自动语音识别（ASR）、口语语言识别（LID）、语音情感识别（SER）和音频事件检测（AED）。SenseVoice-Small 模型采用非自回归端到端框架，具有非常低的推理延迟。它支持 50 多种语言的多语言语音识别，在中文和粤语识别方面表现优于 Whisper 模型。此外，它还具有出色的情感识别和音频事件检测能力。该模型处理 10 秒音频仅需 70 毫秒，比 Whisper-Large 快 15 倍'
      },
      {
        model: 'Kwai-Kolors/Kolors',
        description: 'Kolors 是由快手 Kolors 团队开发的基于潜在扩散的大规模文本到图像生成模型。该模型通过数十亿文本-图像对的训练，在视觉质量、复杂语义准确性以及中英文字符渲染方面展现出显著优势。它不仅支持中英文输入，在理解和生成中文特定内容方面也表现出色'
      }
    ],
    message: 'success',
    redirect_url: '',
    toast: 0,
    type: 'success'
  })
})

// 定义验证Schema
const ChatMessageSchema = z.object({
  role: z.enum(['user', 'assistant', 'system']),
  content: z.string().min(1, '消息内容不能为空')
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
  validator('json', (value,c) => {
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
        baseURL: "https://api.siliconflow.cn/v1"
      });

      const body = c.req.valid('json')
    
      if (body.stream) {
        const response = await client.chat.completions.create({
          model: body.model,
          messages: body.messages,
          stream: body.stream,
          temperature: body.temperature,
          max_tokens: body.max_tokens,
          top_p: body.top_p,
        });

        const stream = new ReadableStream({
          async start(controller) {
            try {
              for await (const chunk of response) {
                const chunkText = `data: ${JSON.stringify(chunk)}\n\n`;
                controller.enqueue(new TextEncoder().encode(chunkText));
              }
              controller.enqueue(new TextEncoder().encode('data: [DONE]\n\n'));
              controller.close();
            } catch (error) {
              controller.error(error);
            }
          }
        });
  
        return new Response(stream, {
          headers: {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
          }
        });
      } else {
        const response = await client.chat.completions.create({
          model: body.model,
          messages: body.messages,
          stream: body.stream,
          temperature: body.temperature,
          max_tokens: body.max_tokens,
          top_p: body.top_p,
        });
        
        return c.json({
          code: 200,
          data: response,
          message: 'success',
          redirect_url: '',
          toast: 0,
          type: 'success'
        })
      }
    } catch (error) {
      log('Error:', error);
      return c.json({ error: '服务器内部错误' }, 500);
    }
  }
)

app.post('/transcriptions', async (c) => {
  const formData = await c.req.formData();

  formData.append("model", "FunAudioLLM/SenseVoiceSmall");
  
  const options = {
    method: 'POST',
    headers: { Authorization: `Bearer ${c.env.API_KEY}` },
    body: formData
  };
  
  return fetch('https://api.siliconflow.cn/v1/audio/transcriptions', options)  
})

export default app