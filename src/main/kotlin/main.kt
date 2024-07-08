package dev.buescher.ai

fun main() {
	neuralNet()
}

// Neural Net training example
fun neuralNet() {
	val net = MLP(3, 4, 4, 1)

	val xs = listOf(
		listOf(2, 3, -1),
		listOf(3, -1, 0.5),
		listOf(0.5, 1, 1),
		listOf(1, 1, -1),
	).map { it.map(::Value) }

	val ys = listOf(1, -1, -1, 1)

	for (k in 0..<100000) {
		// forward pass
		val yPred = xs.map { net(it).first() }
		val loss = ys.zip(yPred)
			.map { (expected, actual) -> (actual - expected).pow(2) }
			.reduce(Value::plus)

		// backward pass
		net.zeroGrad()
		loss.backward()

		// update
		net.parameters().forEach {
			it.data -= 0.15f * it.grad
		}

		println("$k ${String.format("%.7f", loss.data)}")
	}
}
