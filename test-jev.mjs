import { experimental_evaluate as evaluate } from 'ai';

if (!process.env.AI_GATEWAY_API_KEY) {
  console.error('Please set AI_GATEWAY_API_KEY.');
  process.exit(1);
}

const state = process.argv[2] ?? 'The support agent issued a full refund to the customer.';
const started = performance.now();
try {
  const result = await evaluate({
    model: 'typesafe-ai/jev',
    state,
    questions: {
      refunded: { type: 'boolean', instructions: 'Was a refund issued?' },
    },
    maxRetries: 0,
    abortSignal: AbortSignal.timeout(30000),
  });
  console.log(JSON.stringify({ state, elapsedMs: Math.round(performance.now() - started), ...result }, null, 2));
} catch (error) {
  console.error(JSON.stringify({ name: error.name, message: error.message, statusCode: error.statusCode }, null, 2));
  process.exitCode = 1;
}
