import assert from 'node:assert/strict';
import {
  bendFrom,
  clamp,
  formatClock,
  median,
  severityFor,
  smooth,
  uprightPercent,
} from './posture.ts';

assert.equal(clamp(50, 0, 45), 45);
assert.equal(clamp(-2, 0, 45), 0);
assert.equal(median([3, 1, 2]), 2);
assert.equal(median([4, 1, 3, 2]), 2.5);
assert.equal(median([]), 0);

assert.equal(smooth(null, 10), 10);
assert.equal(smooth(10, 20, 0.5), 15);

assert.equal(bendFrom(null, 12), 0);
assert.equal(bendFrom(-3, 9), 12);
assert.equal(bendFrom(5, 5), 0);

assert.equal(severityFor(15, 20), 'good');
assert.equal(severityFor(20, 20), 'good');
assert.equal(severityFor(27, 20), 'warn');
assert.equal(severityFor(28, 20), 'warn');
assert.equal(severityFor(28.5, 20), 'bad');

assert.equal(uprightPercent(0, 0), 0);
assert.equal(uprightPercent(30, 120), 25);
assert.equal(uprightPercent(200, 100), 100);

assert.equal(formatClock(0), '0:00');
assert.equal(formatClock(61_000), '1:01');
assert.equal(formatClock(3_725_000), '1:02:05');

console.log('posture checks passed');
