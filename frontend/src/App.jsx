/**
 * The whole page: pick or type an opening, and watch the model continue it.
 */

import { useEffect, useRef, useState } from 'react'
import { fetchHealth, streamText } from './api'
import { PRESET_PROMPTS } from './prompts'

const STATUS_MESSAGES = {
  checking: 'Checking the model…',
  loaded: 'Model loaded. Pick an opening or write your own.',
  missing: 'No trained model yet. Run training, then restart the API.',
  offline: 'Cannot reach the API. Is it running on port 8100?',
}

export default function App() {
  const [modelStatus, setModelStatus] = useState('checking')
  const [prompt, setPrompt] = useState(PRESET_PROMPTS[0].text)
  const [temperature, setTemperature] = useState(0.8)
  const [length, setLength] = useState(500)
  const [storyPrompt, setStoryPrompt] = useState('')
  const [storyText, setStoryText] = useState('')
  const [isWriting, setIsWriting] = useState(false)
  const [error, setError] = useState(null)
  const sourceRef = useRef(null)

  useEffect(() => {
    fetchHealth()
      .then((health) => setModelStatus(health.model))
      .catch(() => setModelStatus('offline'))
  }, [])

  // Stop the model if the page goes away halfway through a story.
  useEffect(() => {
    return () => sourceRef.current?.close()
  }, [])

  function handleGenerate() {
    setStoryPrompt(prompt)
    setStoryText('')
    setError(null)
    setIsWriting(true)
    sourceRef.current = streamText(prompt, temperature, length, {
      onText: (piece) => setStoryText((current) => current + piece),
      onDone: () => setIsWriting(false),
      onError: () => {
        setIsWriting(false)
        setError('The model stopped unexpectedly. Check that the API is running.')
      },
    })
  }

  function handleStop() {
    sourceRef.current?.close()
    setIsWriting(false)
  }

  return (
    <main className="page">
      <header>
        <h1>Shakespeare Writer</h1>
        <p className="tagline">
          A 1.3M-parameter language model, trained on Shakespeare on a laptop CPU.
        </p>
        <p className={`status status-${modelStatus}`} role="status">
          {STATUS_MESSAGES[modelStatus]}
        </p>
      </header>

      <section className="controls">
        <div className="presets" role="group" aria-label="Openings">
          {PRESET_PROMPTS.map((preset) => (
            <button
              key={preset.label}
              type="button"
              onClick={() => setPrompt(preset.text)}
              disabled={isWriting}
            >
              {preset.label}
            </button>
          ))}
        </div>

        <label htmlFor="prompt">Opening</label>
        <textarea
          id="prompt"
          value={prompt}
          onChange={(event) => setPrompt(event.target.value)}
          maxLength={200}
          rows={3}
          disabled={isWriting}
        />

        <label htmlFor="temperature">
          Creativity <output htmlFor="temperature">{temperature.toFixed(1)}</output>
        </label>
        <input
          id="temperature"
          type="range"
          min="0.1"
          max="1.5"
          step="0.1"
          value={temperature}
          onChange={(event) => setTemperature(Number(event.target.value))}
          disabled={isWriting}
        />

        <label htmlFor="length">
          Length <output htmlFor="length">{length}</output> characters
        </label>
        <input
          id="length"
          type="range"
          min="100"
          max="1500"
          step="100"
          value={length}
          onChange={(event) => setLength(Number(event.target.value))}
          disabled={isWriting}
        />
        <p className="hint">
          The model always finishes the speech it is in, so it may run a little over.
        </p>

        <div className="actions">
          <button
            type="button"
            className="primary"
            onClick={handleGenerate}
            disabled={isWriting || prompt.trim() === ''}
          >
            Generate
          </button>
          <button type="button" onClick={handleStop} disabled={!isWriting}>
            Stop
          </button>
        </div>
      </section>

      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}

      <article className="story" aria-label="Story" aria-busy={isWriting}>
        <span className="story-prompt">{storyPrompt}</span>
        {storyText}
        {isWriting && <span className="cursor" aria-hidden="true" />}
      </article>
    </main>
  )
}
