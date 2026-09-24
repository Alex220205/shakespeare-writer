/**
 * Openings the page offers as one-click prompts.
 *
 * Each is a speaker's name in capitals, a colon and a new line, because that
 * is how the corpus starts every speech. All five speak over a hundred times
 * in Tiny Shakespeare, so the model knows how each of them talks. (Hamlet is
 * not in it at all.)
 */
export const PRESET_PROMPTS = [
  { label: 'Romeo', text: 'ROMEO:\n' },
  { label: 'Juliet', text: 'JULIET:\n' },
  { label: 'Richard III', text: 'KING RICHARD III:\n' },
  { label: 'Petruchio', text: 'PETRUCHIO:\n' },
  { label: 'Queen Elizabeth', text: 'QUEEN ELIZABETH:\n' },
]
