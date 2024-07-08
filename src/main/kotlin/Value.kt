package dev.buescher.ai

import io.github.rchowell.dotlin.*
import java.io.*
import kotlin.math.*

class Value(
	data: Number,
	val op: String = "",
	val children: List<Value> = emptyList(),
) {
	var data: Float = data.toFloat()
	var grad: Float = 0f
	internal var backwardFunction: () -> Unit = {}

	operator fun plus(other: Number) = this + Value(data = other.toDouble())

	operator fun plus(other: Value): Value {
		return Value(data + other.data, "+", listOf(this, other)).also { out ->
			out.backwardFunction = {
				this.grad += out.grad
				other.grad += out.grad
			}
		}
	}

	operator fun minus(other: Number) = this - Value(data = other.toDouble())

	operator fun minus(other: Value): Value {
		return Value(data - other.data, "-", listOf(this, other)).also { out ->
			out.backwardFunction = {
				this.grad += out.grad
				other.grad -= out.grad
			}
		}
	}

	operator fun times(other: Number) = times(Value(data = other.toDouble()))

	operator fun times(other: Value): Value {
		return Value(data * other.data, "*", listOf(this, other)).also { out ->
			out.backwardFunction = {
				this.grad += other.data * out.grad
				other.grad += this.data * out.grad
			}
		}
	}

	operator fun div(other: Number) = this / Value(data = other.toDouble())

	operator fun div(other: Value): Value {
		return Value(data / other.data, "/", listOf(this, other)).also { out ->
			out.backwardFunction = {
				this.grad += out.grad / other.data
				other.grad -= out.grad * this.data / other.data.pow(2)
			}
		}
	}

	operator fun unaryMinus(): Value {
		return Value(-data, "-", listOf(this)).also { out ->
			out.backwardFunction = {
				this.grad -= out.grad
			}
		}
	}

	fun pow(other: Number): Value {
		val x = other.toFloat()
		val power = if (x == round(x)) "^${x.toInt()}" else "^$x"
		return Value(data.pow(x), power, listOf(this)).also { out ->
			out.backwardFunction = {
				this.grad += x * data.pow(x - 1) * out.grad
			}
		}
	}

	fun backward() {
		val result = mutableListOf<Value>()
		val visited = mutableListOf<Value>()

		fun buildResult(value: Value) {
			if (value !in visited) {
				visited.add(value)
				for (child in value.children) {
					buildResult(child)
				}
				result += value
			}
		}

		grad = 1f
		buildResult(this)

		for (node in result.reversed()) {
			node.backwardFunction()
		}
	}

	override fun toString(): String = "Value(data=$data, grad=$grad)"
}

operator fun Number.plus(other: Value) = Value(data = this.toDouble()) + other
operator fun Number.minus(other: Value) = Value(data = this.toDouble()) - other
operator fun Number.times(other: Value) = Value(data = this.toDouble()) * other
operator fun Number.div(other: Value) = Value(data = this.toDouble()) / other

fun exp(value: Value): Value {
	return Value(exp(value.data), "exp", listOf(value)).also { out ->
		out.backwardFunction = {
			value.grad += out.data * out.grad
		}
	}
}

fun tanh(value: Value): Value {
	val t = tanh(value.data)
	return Value(t, "tanh", listOf(value)).also { out ->
		out.backwardFunction = {
			value.grad += (1 - t.pow(2)) * out.grad
		}
	}
}

fun relu(value: Value): Value {
	return Value(max(0f, value.data), "ReLU", listOf(value)).also { out ->
		out.backwardFunction = {
			value.grad += if (out.data > 0) out.grad else 0f
		}
	}
}

fun exportDot(value: Value) {
	fun trace(root: Value): Pair<Set<Value>, Set<Pair<Value, Value>>> {
		val nodes = mutableSetOf<Value>()
		val edges = mutableSetOf<Pair<Value, Value>>()

		fun build(v: Value) {
			if (v !in nodes) {
				nodes.add(v)
				v.children.forEach { child ->
					edges.add(child to v)
					build(child)
				}
			}
		}
		build(root)

		return nodes to edges
	}

	val (nodes, edges) = trace(value)

	val dot = digraph {
		rankdir = DotRankDir.LR

		nodes.forEach { node ->
			val id = System.identityHashCode(node).toString()
			+id + {
				label = String.format("{ data %.4f | grad %.4f }", node.data, node.grad)
				shape = DotNodeShape.RECORD
			}

			if (node.op.isNotEmpty()) {
				+"\"${id + node.op}\"" + { label = node.op }
				"\"${id + node.op}\"" - id
			}
		}

		edges.forEach { (child, parent) ->
			val childId = System.identityHashCode(child).toString()
			val parentId = System.identityHashCode(parent).toString()
			childId - "\"${parentId + parent.op}\""
		}
	}

	val tmp = File("tmp.dot")

	tmp.writer().use { it.write(dot.dot()) }

	ProcessBuilder("dot", "-Tsvg", tmp.path)
		.redirectOutput(File("result.svg"))
		.redirectError(ProcessBuilder.Redirect.INHERIT)
		.start()
		.waitFor()

	tmp.delete()
}
