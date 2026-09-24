/**
 * ESLint flat configuration.
 *
 * Catches the mistakes that are silent in JavaScript until they are not:
 * unused bindings, and in React the hook rules, where a conditional hook call
 * is legal JavaScript and a broken component.
 */

import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'

export default [
  { ignores: ['dist/**', 'node_modules/**'] },
  js.configs.recommended,
  reactHooks.configs.flat['recommended-latest'],
  {
    files: ['**/*.{js,jsx}'],
    languageOptions: {
      ecmaVersion: 'latest',
      sourceType: 'module',
      globals: { ...globals.browser, ...globals.vitest },
      parserOptions: {
        ecmaFeatures: { jsx: true },
      },
    },
  },
]
