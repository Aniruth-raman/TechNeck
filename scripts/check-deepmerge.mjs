import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

// Resolves the installed copy, so this also proves the package.json override is wired up.
const require = createRequire(import.meta.url);
const merge = require('deepmerge');

const nested = JSON.parse('{"session":{"__proto__":{"admin":true}}}');
const poisoned = merge({ session: null }, nested);
assert.equal(poisoned.session.admin, undefined);
assert.equal(Object.getPrototypeOf(poisoned.session), Object.prototype);
assert.equal(Object.hasOwn(poisoned.session, 'admin'), false);

const topLevel = JSON.parse('{"__proto__":{"polluted":true}}');
const result = merge({}, topLevel);
assert.equal(result.polluted, undefined);
assert.equal({}.polluted, undefined);

assert.deepEqual(merge({ a: { b: 1 } }, { a: { c: 2 } }), { a: { b: 1, c: 2 } });
assert.deepEqual(merge({ list: [1] }, { list: [2] }), { list: [1, 2] });
assert.deepEqual(merge({ a: 1 }, { b: 2 }, {}), { a: 1, b: 2 });

console.log('deepmerge guard ok');
