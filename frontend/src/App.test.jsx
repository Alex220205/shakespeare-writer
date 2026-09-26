import { act, fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi } from 'vitest'
import App from './App'
import { FakeEventSource } from './setupTests'

/** The stream the page opened most recently. */
function latestStream() {
  return FakeEventSource.instances.at(-1)
}

/** Send one piece of text down the stream, JSON-encoded as the API does. */
function sendText(stream, text) {
  act(() => stream.emit('message', JSON.stringify(text)))
}

test('says the model is loaded when the API has one', async () => {
  render(<App />)

  expect(await screen.findByText(/model loaded/i)).toBeInTheDocument()
})

test('tells the user to train when the API has no model', async () => {
  globalThis.fetch = vi.fn(async () => ({
    ok: true,
    json: async () => ({ status: 'degraded', model: 'missing', version: '0.1.0' }),
  }))

  render(<App />)

  expect(await screen.findByText(/no trained model yet/i)).toBeInTheDocument()
})

test('a preset opening fills the prompt box', async () => {
  const user = userEvent.setup()
  render(<App />)

  await user.click(screen.getByRole('button', { name: 'Juliet' }))

  expect(screen.getByLabelText('Opening')).toHaveValue('JULIET:\n')
})

test('streams text into the story as it arrives, then allows another', async () => {
  const user = userEvent.setup()
  render(<App />)

  await user.click(screen.getByRole('button', { name: 'Generate' }))
  const stream = latestStream()
  expect(stream.url).toContain('prompt=ROMEO%3A%0A')
  expect(stream.url).toContain('length=500')
  expect(screen.getByRole('button', { name: 'Generate' })).toBeDisabled()

  sendText(stream, 'But')
  sendText(stream, ' soft!')
  expect(screen.getByRole('article', { name: 'Story' })).toHaveTextContent(
    'ROMEO: But soft!',
  )

  act(() => stream.emit('done', '""'))
  expect(stream.closed).toBe(true)
  expect(screen.getByRole('button', { name: 'Generate' })).toBeEnabled()
})

test('the length slider sets how much the model is asked to write', async () => {
  const user = userEvent.setup()
  render(<App />)
  const slider = screen.getByRole('slider', { name: /length/i })

  // A range input has no typing interaction; change is what dragging fires.
  fireEvent.change(slider, { target: { value: '1200' } })
  await user.click(screen.getByRole('button', { name: 'Generate' }))

  expect(screen.getByText('1200')).toBeInTheDocument()
  expect(latestStream().url).toContain('length=1200')
})

test('Stop closes the stream and keeps what was written', async () => {
  const user = userEvent.setup()
  render(<App />)
  await user.click(screen.getByRole('button', { name: 'Generate' }))
  const stream = latestStream()
  sendText(stream, 'What light')

  await user.click(screen.getByRole('button', { name: 'Stop' }))

  expect(stream.closed).toBe(true)
  expect(screen.getByRole('button', { name: 'Generate' })).toBeEnabled()
  expect(screen.getByRole('article', { name: 'Story' })).toHaveTextContent('What light')
})

test('a broken stream shows an error instead of writing forever', async () => {
  const user = userEvent.setup()
  render(<App />)
  await user.click(screen.getByRole('button', { name: 'Generate' }))
  const stream = latestStream()

  act(() => stream.emit('error'))

  expect(screen.getByRole('alert')).toHaveTextContent(/stopped unexpectedly/i)
  expect(stream.closed).toBe(true)
  expect(screen.getByRole('button', { name: 'Generate' })).toBeEnabled()
})
