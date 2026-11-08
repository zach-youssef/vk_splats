// stream_test on the web (Node): the failure exit (every check is in the app)
import * as zgal from './zgal/index.ts';
import { type MainInputs } from './stream_test.zgal.ts';

export function mainInputs(_context: zgal.Context): MainInputs {
  return {
    fail() {
      globalThis.process?.exit(1);
      throw new Error('stream_test failed');
    },
  };
}
