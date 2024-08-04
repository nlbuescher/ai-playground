package dev.buescher.ai

import kotlin.random.*

class Vector(
	internal val data: Array<Value>,
	internal val offset: Int,
	val size: Int,
) : Iterable<Value> {
	constructor(data: Array<Value>) : this(data, 0, data.size)

	operator fun get(index: Int): Value {
		check(index in 0..<size) { "out of bounds" }
		return data[offset + index]
	}

	operator fun div(value: Number) = this / Value(value)
	operator fun div(value: Value): Vector {
		val newData = data.copyOfRange(offset, offset + size)
		newData.indices.forEach { i ->
			newData[i] /= value
		}
		return Vector(newData, 0, size)
	}

	override fun iterator(): Iterator<Value> {
		return object : Iterator<Value> {
			private var index = 0

			override fun hasNext(): Boolean = index < size
			override fun next(): Value = data[offset + index].also { index += 1 }
		}
	}

	override fun toString(): String = joinToString(", ", "[", "]") { "${it.data}" }
}

fun Vector.sum(): Value {
	if (size == 0) return Value(0)
	return reduce { acc, value -> acc + value }
}

fun multinomial(input: Vector, sampleCount: Int, generator: Random = Random.Default): List<Int> {
	val cumulativeProbabilities = input.scan(0f) { acc, it -> acc + it.data }.drop(1)

	return List(sampleCount) {
		val rand = generator.nextFloat()
		cumulativeProbabilities.indexOfFirst { it > rand }
	}
}