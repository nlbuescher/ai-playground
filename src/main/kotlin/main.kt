package dev.buescher.ai

import java.io.*
import kotlin.random.*

fun main() {
	val t = Tensor(listOf(1))
	println(t.shape)
}

fun languageModeling() {
	val words = FileReader("names.txt").readLines()

	val N = Matrix(27, 27)
	val chars = words.joinToString("")
		.toCharArray()
		.toSet()
		.sorted()
		.let { listOf('.') + it }

	val stoi = chars.mapIndexed { i, it -> it to i }.toMap()
	val itos = stoi.map { (it, i) -> i to it }.toMap()

	words.forEach { word ->
		val c = listOf('.') + word.toCharArray().asList() + listOf('.')
		c.windowed(2).forEach { (c1, c2) ->
			N[stoi.getValue(c1), stoi.getValue(c2)] += 1
		}
	}

//	val P = N / N.sumColumns()

	val generator = Random(2147483647)

	repeat(20) {
		val out = mutableListOf<Char>()
		var i = 0
		while (true) {
			val p = N[i]
			i = multinomial(p, sampleCount = 1, generator).first()
			if (i == 0) {
				break
			}
			else {
				out.add(itos[i]!!)
			}
		}

		println(out.joinToString(""))
	}
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
