// densify_test on the web (Node): the failure exit (every check is in the app)
import * as zgal from './zgal/index.ts';
import { type MainInputs } from './densify_test.zgal.ts';

export function mainInputs(_context: zgal.Context): MainInputs {
  return {
    fail() {
      globalThis.process?.exit(1);
      throw new Error('densify_test failed');
    },
  };
}
