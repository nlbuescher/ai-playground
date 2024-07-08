package dev.buescher.ai

import kotlin.random.*

interface Module {
	fun parameters(): List<Value>

	fun zeroGrad() {
		parameters().forEach { it.grad = 0f }
	}
}

class Neuron(
	inCount: Int,
	private val isNonLinear: Boolean = true
) : Module {
	private val weights = (0..<inCount).map { Value(Random.nextDouble(-1.0, 1.0)) }
	private val bias = Value(Random.nextDouble(-1.0, 1.0))

	operator fun invoke(x: List<Value>): Value {
		check(weights.size == x.size) { "the input must have the same size as the neuron" }

		return weights.asSequence()
			.zip(x.asSequence())
			.map { (wi, xi) -> wi * xi }
			.reduce(Value::plus)
			.let { tanh(it) } + bias
	}

	override fun parameters() = weights + bias

	override fun toString() = "${if (isNonLinear) "ReLU" else "Linear"}Neuron(${weights.size})"
}

class Layer(inCount: Int, outCount: Int) : Module {
	private val neurons = (0..<outCount).map { Neuron(inCount) }

	operator fun invoke(x: List<Value>): List<Value> {
		return neurons.map { it(x) }
	}

	override fun parameters() = neurons.flatMap { it.parameters() }

	override fun toString(): String = "Layer of [${neurons.joinToString(", ")}]"
}

class MLP(inCount: Int, firstOutCount: Int, vararg otherOutCounts: Int) : Module {
	private val layers: List<Layer>

	init {
		val sizes = listOf(inCount, firstOutCount) + otherOutCounts.asList()
		layers = sizes.windowed(2, 1).map { (inCount, outCount) -> Layer(inCount, outCount) }
	}

	operator fun invoke(x: List<Value>): List<Value> {
		return layers.fold(x) { acc, layer -> layer(acc) }
	}

	override fun parameters(): List<Value> = layers.flatMap { it.parameters() }

	override fun toString() = "MLP of [${layers.joinToString(", ")}]"
}
