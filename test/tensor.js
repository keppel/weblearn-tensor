const test = require('tape')
const Tensor = require('../index.js')

test('matrix-vector multiplication arity 2', t=>{
  t.plan(1)

  const x = Tensor(3).fill(0)
  const M = Tensor(3, 2).fill(3)
  const y = Tensor(2).fill(2)

  x.addmv(M, y)
  t.deepEqual(x.values, [12, 12, 12])
})

test('matrix-vector multiplication arity 3', t=>{
  t.plan(3)

  const x = Tensor(3).fill(0)
  const M = Tensor(3, 2).fill(3)
  const y = Tensor(2).fill(2)

  x.addmv(0, M, y)
  t.deepEqual(x.values, [0, 0, 0])

  x.addmv(1, M, y)
  t.deepEqual(x.values, [12, 12, 12])

  x.addmv(2, M, y)
  t.deepEqual(x.values, [36, 36, 36])
})

test('matrix-vector multiplication arity 4', t=>{
  t.plan(3)

  const x = Tensor(3).fill(5)
  const M = Tensor(3, 2).fill(3)
  const y = Tensor(2).fill(2)

  x.addmv(0, 0, M, y)
  t.deepEqual(x.values, [0, 0, 0])

  x.addmv(1, 1, M, y)
  t.deepEqual(x.values, [12, 12, 12])

  x.addmv(2, 2, M, y)
  t.deepEqual(x.values, [48, 48, 48])
})

test('vector outer-product (addr)', t=>{
  t.plan(2)
  const x = Tensor([1, 2, 3])
  const y = Tensor([1, 2])
  const M = Tensor(3, 2).zero()

  M.addr(x, y)
  t.deepEqual(M.values, [[1, 2], [2, 4], [3, 6]])

  M.addr(2, 1, x, y)
  t.deepEqual(M.values, [[3, 6], [6, 12], [9, 18]])

})


test('matrix transposition', t=>{
  t.plan(2)
  const x = Tensor([[1, 2], [3, 4], [5, 6]])

  t.deepEqual(x.t().values, [[1, 3, 5], [2, 4, 6]])
  // but make sure the original tensor isn't changed
  t.deepEqual(x.values, [[1, 2], [3, 4], [5, 6]])

})

test('get tensor size', t =>{
  t.plan(2)
  const x = Tensor(2, 3)
  t.ok(x.size(0) === 2)
  t.ok(x.size(1) === 3)
})

test('in-place sigmoid', t =>{
  t.plan(1)
  const x = Tensor([.25, .5, 1, -2])
  x.sigmoid()
  const truncatedValues = x.values.map(v => Number(v.toFixed(2)))
  t.deepEqual(truncatedValues, [.56, .62, .73, .12])
})

test('in-place tanh', t =>{
  t.plan(1)
  const x = Tensor([.25, .5, 1, -2])
  x.tanh()
  const truncatedValues = x.values.map(v => Number(v.toFixed(2)))
  t.deepEqual(truncatedValues, [.24, .46, .76, -.96])
})

test('element-wise tensor multiplication (cmul)', t =>{
  t.plan(1)
  let x = Tensor(2, 2).fill(2)
  let y = Tensor(4).fill(3)
  x.cmul(y)
  t.deepEqual(x.values, [[6, 6], [6, 6]])
})

test('tensor views', t =>{
  t.plan(1)
  const x = Tensor([1, 2, 3, 4])
  const y = x.view(2, 2)
  t.equal(y.values[1][1], 4)
  // todo: test that memory manipulation of original affects the view and vice versa
})



