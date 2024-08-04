package dev.buescher.ai

import kotlin.math.*

class Matrix(
	private val rowCount: Int,
	private val colCount: Int,
	init: (Int, Int) -> Value = { _, _ -> Value(0) },
) {
	private val data: Array<Value> = Array(rowCount * colCount) { i -> init(i / colCount, i % colCount) }

	operator fun get(row: Int) = Vector(data, row * colCount, colCount)

	operator fun get(row: Int, col: Int) = data[row * colCount + col]
	operator fun set(row: Int, col: Int, value: Value) = run { data[row * colCount + col] = value }
	operator fun set(row: Int, col: Int, value: Number) = run { data[row * colCount + col] = Value(value) }

	operator fun div(vector: Vector): Matrix {
		check(vector.size == rowCount) { "Size mismatch: Vector[${vector.size}] vs Matrix[${rowCount}, ${colCount}]" }

		return Matrix(rowCount, colCount) { row, col ->
			data[row * colCount + col] / vector[row]
		}
	}

	override fun toString(): String {
		val strings = data.map { "${it.data}" }.chunked(colCount)
		val colWidths = IntArray(colCount) { 0 }
		for (row in 0..<rowCount) {
			for (col in 0..<colCount) {
				colWidths[col] = max(colWidths[col], strings[row][col].length)
			}
		}
		return strings
			.map { row -> row.mapIndexed { i, it -> i to it } }
			.joinToString(",\n ", "[", "]") {
				it.joinToString(", ", "[", "]") { (col, it) ->
					it.padStart(colWidths[col])
				}
			}
	}
}