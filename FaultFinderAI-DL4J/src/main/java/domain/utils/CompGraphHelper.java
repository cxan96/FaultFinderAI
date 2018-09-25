package domain.utils;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationReLU;

public class CompGraphHelper {

	public static ComputationGraphConfiguration.GraphBuilder addLayers(
			ComputationGraphConfiguration.GraphBuilder graphBuilder, int layerNumber, int filterXSize, int filterYSize,
			int filterXStride, int filterYStride, int nIn, int nOut, int poolXSize, int poolYSize, int poolXStride,
			int poolYStride) {
		String input = "maxpooling2d_" + (layerNumber - 1);
		if (!graphBuilder.getVertices().containsKey(input)) {
			input = "activation_" + (layerNumber - 1);
		}
		if (!graphBuilder.getVertices().containsKey(input)) {
			input = "concatenate_" + (layerNumber - 1);
		}
		if (!graphBuilder.getVertices().containsKey(input)) {
			input = "input";
		}

		return addLayers(graphBuilder, layerNumber, input, filterXSize, filterYSize, filterXStride, filterYStride, nIn,
				nOut, poolXSize, poolYSize, poolXStride, poolYStride);
	}

	public static ComputationGraphConfiguration.GraphBuilder addLayers(
			ComputationGraphConfiguration.GraphBuilder graphBuilder, int layerNumber, String input, int filterXSize,
			int filterYSize, int filterXStride, int filterYStride, int nIn, int nOut, int poolXSize, int poolYSize,
			int poolXStride, int poolYStride) {
		graphBuilder
				.addLayer("convolution2d_" + layerNumber,
						new ConvolutionLayer.Builder(filterXSize, filterYSize).nIn(nIn).nOut(nOut)
								.weightInit(WeightInit.XAVIER).convolutionMode(ConvolutionMode.Same).hasBias(false)
								.stride(filterXStride, filterYStride).activation(Activation.IDENTITY).build(),
						input)
				.addLayer("batchnormalization_" + layerNumber,
						new BatchNormalization.Builder().nIn(nOut).nOut(nOut).weightInit(WeightInit.XAVIER)
								.activation(Activation.IDENTITY).build(),
						"convolution2d_" + layerNumber)
				.addLayer("activation_" + layerNumber,
						new ActivationLayer.Builder().activation(new ActivationReLU()).build(),
						"batchnormalization_" + layerNumber);
		if (poolXSize > 0 || poolYSize > 0) {
			graphBuilder.addLayer(
					"maxpooling2d_" + layerNumber, new SubsamplingLayer.Builder().kernelSize(poolXSize, poolYSize)
							.stride(poolXStride, poolYStride).convolutionMode(ConvolutionMode.Same).build(),
					"activation_" + layerNumber);
		}
		// ActivationLReLU(0.1)
		return graphBuilder;
	}
}
