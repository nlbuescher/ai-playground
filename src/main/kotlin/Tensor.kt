package dev.buescher.ai

class Tensor(data: Any) {
	val shape = calculateShape(data)
	private val data = flatten(data, this.shape).toList()

	private fun toStringRecursive(
		data: List<String>,
		shape: List<Int>,
		depth: Int = 0,
		indent: Boolean = false,
	): String = when (shape.size) {
		0 -> data.first().toString()
		1 -> data.toString()
		2 -> buildString {
			val (_, chunkSize) = shape
			val chunks = data.chunked(chunkSize)

			append(
				chunks.joinToString(
					separator = ",\n${" ".repeat(depth + 1)}",
					prefix = if (indent) "[".padStart(depth + 1) else "[",
					postfix = "]",
				) {
					it.joinToString(separator = ", ", prefix = "[", postfix = "]")
				},
			)
		}

		else -> {
			val n = shape.first()
			val subSize = data.size / n
			val subShape = shape.drop(1)

			val parts = data.chunked(subSize).mapIndexed { i, chunk ->
				toStringRecursive(chunk, subShape, depth + 1, i % n != 0)
			}

			parts.joinToString(
				separator = ",".padEnd(subShape.size + 1, '\n'),
				prefix = if (indent) "[".padStart(depth + 1) else "[",
				postfix = "]",
			)
		}
	}

	override fun toString(): String {
		val strings = data.map(Number::toString)
		val colWidth = strings.maxOf { it.length }
		return "Tensor(${toStringRecursive(strings.map { it.padStart(colWidth) }, shape, depth = 7)})"
	}

	companion object {
		@OptIn(ExperimentalUnsignedTypes::class)
		@JvmStatic
		private fun calculateShape(data: Any?): List<Int> = when (data) {
			is ByteArray -> listOf(data.size)
			is UByteArray -> listOf(data.size)
			is ShortArray -> listOf(data.size)
			is UShortArray -> listOf(data.size)
			is IntArray -> listOf(data.size)
			is UIntArray -> listOf(data.size)
			is LongArray -> listOf(data.size)
			is ULongArray -> listOf(data.size)
			is FloatArray -> listOf(data.size)
			is DoubleArray -> listOf(data.size)
			is Array<*> -> listOf(data.size) + calculateShape(data.first())
			is Collection<*> -> listOf(data.size) + calculateShape(data.first())
			else -> emptyList()
		}

		@OptIn(ExperimentalUnsignedTypes::class)
		@JvmStatic
		private fun flatten(data: Any?, shape: List<Int>, depth: Int = 0): List<Number> {
			val dimSize = shape.firstOrNull() ?: 0
			return when (data) {
				null -> throw IllegalArgumentException("unexpected null")
				is Number -> listOf(data)
				is ByteArray -> data.asList()
				is UByteArray -> data.asByteArray().asList()
				is ShortArray -> data.asList()
				is UShortArray -> data.asShortArray().asList()
				is IntArray -> data.asList()
				is UIntArray -> data.asIntArray().asList()
				is LongArray -> data.asList()
				is ULongArray -> data.asLongArray().asList()
				is FloatArray -> data.asList()
				is DoubleArray -> data.asList()
				is Array<*> -> {
					check(data.size == dimSize) {
						"expected sequence of length $dimSize at dim $depth (got ${data.size})"
					}
					data.flatMap { flatten(it, shape.drop(1), depth + 1) }
				}

				is Collection<*> -> {
					check(data.size == dimSize) {
						"expected sequence of length $dimSize at dim $depth (got ${data.size})"
					}
					data.flatMap { flatten(it, shape.drop(1), depth + 1) }
				}

				else -> throw IllegalArgumentException("unexpected type ${data::class}")
			}
		}
	}
}