// splat_view on the web: the failure exit (every check is in the app)
import * as zgal from './zgal/index.ts';
import { type MainInputs } from './splat_view.zgal.ts';

export function mainInputs(_context: zgal.Context): MainInputs {
  return {
    fail() {
      globalThis.process?.exit(1);
      throw new Error('splat_view failed');
    },
  };
}
