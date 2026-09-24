/**
 * Wrappers around fetch and EventSource for the shakespeare-writer API.
 *
 * Components ask for data here rather than assembling URLs, so the base URL
 * and the error handling live in one place.
 */

// Vite replaces import.meta.env.VITE_API_URL at build time. The fallback is
// the API's address when run locally with uvicorn's defaults.
const BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

// fetch has no default timeout. A server that accepts the connection and never
// answers would leave the status line saying "Checking…" forever.
const TIMEOUT_MS = 5000

/**
 * Ask the API whether it is up and has a trained model.
 *
 * @returns {Promise<{status: string, model: string, version: string}>}
 * @throws {Error} If the API cannot be reached or answers with an error.
 */
export async function fetchHealth() {
  const response = await fetch(`${BASE_URL}/health`, {
    signal: AbortSignal.timeout(TIMEOUT_MS),
  })
  if (!response.ok) {
    throw new Error(`GET /health failed: ${response.status}`)
  }
  return response.json()
}

/**
 * Stream the model's continuation of a prompt.
 *
 * The API sends one event per token, then a "done" event. The stream must be
 * closed on "done": EventSource treats a stream that ends as a dropped
 * connection and reconnects, which here would ask for a second story.
 *
 * @param {string} prompt Text for the model to continue.
 * @param {number} temperature 0.1 to 1.5; higher is more inventive.
 * @param {object} handlers
 * @param {(text: string) => void} handlers.onText Called with each new piece.
 * @param {() => void} handlers.onDone Called once the model has finished.
 * @param {() => void} handlers.onError Called if the stream cannot be opened
 *   or breaks. The browser does not say why, so neither can this.
 * @returns {EventSource} Close it to stop the model early.
 */
export function streamText(prompt, temperature, { onText, onDone, onError }) {
  const query = new URLSearchParams({ prompt, temperature: String(temperature) })
  const source = new EventSource(`${BASE_URL}/generate?${query}`)

  // Each piece is JSON-encoded on the wire so newlines and leading spaces
  // survive the event-stream format.
  source.addEventListener('message', (event) => {
    onText(JSON.parse(event.data))
  })
  source.addEventListener('done', () => {
    source.close()
    onDone()
  })
  source.addEventListener('error', () => {
    source.close()
    onError()
  })

  return source
}
