plugins {
	kotlin("jvm") version "2.0.0"
}

group = "dev.buescher.ai_playground"

repositories {
	mavenCentral()
}

dependencies {
	implementation(kotlin("reflect"))
	implementation("io.github.rchowell:dotlin:+")
	testImplementation(kotlin("test"))
}

kotlin {
	jvmToolchain(21)
}
