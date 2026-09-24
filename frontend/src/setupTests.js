/**
 * Vitest setup, run before every test file.
 *
 * Registers jest-dom's matchers, and stubs the two browser APIs this page uses
 * to reach the server. Both are stubbed for the whole suite, before any test
 * runs, so no test can pass or fail on whether an API happens to be running.
 */

import { beforeEach, vi } from 'vitest'

import '@testing-library/jest-dom/vitest'

/**
 * Stands in for the browser's EventSource. Tests take the latest instance
 * from FakeEventSource.instances and call emit() to play the server's part.
 */
export class FakeEventSource {
  static instances = []

  constructor(url) {
    this.url = url
    this.listeners = {}
    this.closed = false
    FakeEventSource.instances.push(this)
  }

  addEventListener(type, listener) {
    if (!this.listeners[type]) {
      this.listeners[type] = []
    }
    this.listeners[type].push(listener)
  }

  close() {
    this.closed = true
  }

  /** Deliver an event to the page, as the browser would. */
  emit(type, data) {
    for (const listener of this.listeners[type] ?? []) {
      listener({ data })
    }
  }
}

globalThis.EventSource = FakeEventSource

beforeEach(() => {
  FakeEventSource.instances = []
  // A healthy API with a trained model, unless a test says otherwise.
  globalThis.fetch = vi.fn(async () => ({
    ok: true,
    json: async () => ({ status: 'ok', model: 'loaded', version: '0.1.0' }),
  }))
})
