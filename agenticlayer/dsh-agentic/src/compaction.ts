import { BasicCompactionEngine } from '@deepseek-ai/dsh-compaction-basic'
import type { Agent } from '@deepseek-ai/dsh-agent'
import type { SummarizationInput, SummaryResult } from '@deepseek-ai/dsh-compaction-basic/src/summarizer.ts'

/** Keeps native range validation, transactional replacement, and durable log. */
export default class FixtureCompactionEngine extends BasicCompactionEngine {
  protected override async summarize(input: SummarizationInput, agent: Agent, signal?: AbortSignal): Promise<SummaryResult> {
    signal?.throwIfAborted()
    const fixed = process.env.AGENTIC_FIXED_SUMMARY
    if (fixed === undefined) return super.summarize(input, agent, signal)
    if (!fixed.trim()) throw new Error('AGENTIC_FIXED_SUMMARY must be nonempty')
    const route = agent.session.requestHeader()?.config
    const provider = route?.provider ?? agent.options.provider
    const model = route?.model ?? agent.options.model
    if (!provider || !model) throw new Error('compaction needs a routed provider and model')
    const rawOutput = [{ type: 'text' as const, text: fixed }]
    return { summary: rawOutput, rawOutput, provider, model, maxTokens: 256 }
  }
}
