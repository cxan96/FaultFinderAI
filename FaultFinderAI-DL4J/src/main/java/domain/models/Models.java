package domain.models;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex.Op;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.Upsampling2D;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.BaseActivationFunction;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.activations.impl.ActivationLReLU;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;

import utils.FaultUtils;

/**
 * This class is used to retrieve all the different models that are available.
 */
public class Models {

	public static ComputationGraph DriftChamber(int height, int width, int numChannels, int numClasses,
			double[][] priorBoxes) {
		int nBoxes = priorBoxes.length;
		INDArray priors = Nd4j.create(priorBoxes);

		double lambdaNoObj = 0.5;
		double lambdaCoord = 1.0;

		double learningRate = 1e-4;
		// goes on AdamUpdater
		// .learningRate(learningRate)
		double l2Rate = 0.0001;
		int seed = 123;
		double leakyReLUVaue = 0.01;

		GraphBuilder graphBuilder = new NeuralNetConfiguration.Builder().seed(seed)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.gradientNormalization(GradientNormalization.RenormalizeL2PerLayer).gradientNormalizationThreshold(1.0)
				.weightInit(WeightInit.XAVIER).updater(new Adam.Builder().learningRate(learningRate).build()).l2(l2Rate)
				.activation(Activation.IDENTITY).graphBuilder().addInputs("input")
				.setInputTypes(InputType.convolutional(height, width, numChannels))
				.addLayer("cnn0", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(32)
						.activation(new ActivationLReLU(leakyReLUVaue)).convolutionMode(ConvolutionMode.Same).build(),
						"input");
		resBlock(graphBuilder, "cnn0", 1, new ActivationLReLU(leakyReLUVaue), 64, 128, 64);
		resBlock(graphBuilder, ("activationBlock" + 1), 2, new ActivationLReLU(leakyReLUVaue), 128, 256, 128);
		/**
		 * set up yolo
		 */

		graphBuilder
				/**
				 * I need this to make the width an odd number
				 */
				.addLayer("cnn3", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(512)
						.activation(new ActivationLReLU(leakyReLUVaue)).convolutionMode(ConvolutionMode.Same).build(),
						"activationBlock" + 2)
				//
				.addLayer("cnn4", new ConvolutionLayer.Builder(1, 1).stride(1, 1).nOut(512)
						.activation(new ActivationLReLU(leakyReLUVaue)).convolutionMode(ConvolutionMode.Same).build(),
						"cnn3")

				.addLayer("up1", new Upsampling2D.Builder().size(new int[] { 2, 2 }).build(), "cnn4")

				.addVertex("merge1", new MergeVertex(), "up1", "downSamplecnnBlock" + 1)
				// another block
				.addLayer("cnn5", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(256)
						.activation(new ActivationLReLU(leakyReLUVaue)).convolutionMode(ConvolutionMode.Same).build(),
						"merge1")
				//
				.addLayer("cnn6", new ConvolutionLayer.Builder(1, 1).stride(1, 1).nOut(256)
						.activation(new ActivationLReLU(leakyReLUVaue)).convolutionMode(ConvolutionMode.Same).build(),
						"cnn5")
				.addLayer("up2", new Upsampling2D.Builder().size(new int[] { 2, 2 }).build(), "cnn6")

				.addVertex("merge2", new MergeVertex(), "up2", "cnn0")

				.addLayer("cnn7", new ConvolutionLayer.Builder(1, 1).stride(1, 1).nOut(128)
						.activation(new ActivationLReLU(leakyReLUVaue)).convolutionMode(ConvolutionMode.Same).build(),
						"merge2")
				.addLayer("cnn8", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(128)
						.activation(new ActivationLReLU(leakyReLUVaue)).convolutionMode(ConvolutionMode.Same).build(),
						"cnn7")
				.addLayer("cnn9", new ConvolutionLayer.Builder(1, 1).stride(1, 1).nOut(64)
						.activation(new ActivationLReLU(leakyReLUVaue)).convolutionMode(ConvolutionMode.Same).build(),
						"cnn8")
				.addLayer("cnn10",
						new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(64)
								.activation(new ActivationLReLU(leakyReLUVaue)).build(),
						"cnn9")
				.addLayer("cnn11",
						new ConvolutionLayer.Builder(2, 2).stride(1, 1).nOut(128)
								.activation(new ActivationLReLU(leakyReLUVaue)).padding(1, 1).build(),
						"cnn10")
				.addLayer("cnnBeforeYolo",
						new ConvolutionLayer.Builder(1, 1).nOut(nBoxes * (5 + numClasses)).weightInit(WeightInit.XAVIER)
								.stride(1, 1).convolutionMode(ConvolutionMode.Same).activation(new ActivationIdentity())
								.build(),
						"cnn11")
				.addLayer("outputs", new Yolo2OutputLayer.Builder().lambbaNoObj(lambdaNoObj).lambdaCoord(lambdaCoord)
						.boundingBoxPriors(priors).build(), "cnnBeforeYolo")
				.setOutputs("outputs");

		ComputationGraph neuralNetwork = new ComputationGraph(graphBuilder.build());

		// initialize the network
		neuralNetwork.init();
		// System.out.println(neuralNetwork.summary(InputType.convolutional(height,
		// width, numChannels)));

		return neuralNetwork;

	}

	public static ComputationGraph KunkelPetersUYolo4SL(int height, int width, int numChannels, int numClasses,
			double[][] priorBoxes) {

		int nBoxes = priorBoxes.length;

		INDArray priors = Nd4j.create(priorBoxes);

		double lambdaNoObj = 0.5;
		double lambdaCoord = 1.0;

		double learningRate = 1e-4;
		// goes on AdamUpdater
		// .learningRate(learningRate)
		double l2Rate = 0.0001;
		int seed = 123;
		double leakyReLUVaue = 0.01;

		GraphBuilder graphBuilder = new NeuralNetConfiguration.Builder().seed(seed)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.gradientNormalization(GradientNormalization.RenormalizeL2PerLayer).gradientNormalizationThreshold(1.0)
				.weightInit(WeightInit.XAVIER).updater(new Adam.Builder().learningRate(learningRate).build()).l2(l2Rate)
				.activation(Activation.IDENTITY).graphBuilder().addInputs("input")
				.setInputTypes(InputType.convolutional(height, width, numChannels))
				.addLayer("cnn1", new ConvolutionLayer.Builder(2, 3).stride(1, 1).nOut(64)
						.activation(new ActivationLReLU(leakyReLUVaue)).convolutionMode(ConvolutionMode.Same).build(),
						"input")
				.addLayer("cnn2", new ConvolutionLayer.Builder(2, 2).stride(1, 1).nOut(64)
						.activation(new ActivationLReLU(leakyReLUVaue)).convolutionMode(ConvolutionMode.Same).build(),
						"cnn1")
				.addLayer("pool1",
						new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2)
								.build(),
						"cnn2")
				.addLayer("cnn3", new ConvolutionLayer.Builder(2, 3).stride(1, 1).nOut(128)
						.activation(new ActivationLReLU(leakyReLUVaue)).convolutionMode(ConvolutionMode.Same).build(),
						"pool1")
				//
				.addLayer("cnn4", new ConvolutionLayer.Builder(2, 2).stride(1, 1).nOut(128)
						.activation(new ActivationLReLU(leakyReLUVaue)).convolutionMode(ConvolutionMode.Same).build(),
						"cnn3")

				.addLayer("up3", new Upsampling2D.Builder().size(new int[] { 2, 2 }).build(), "cnn4")

				.addVertex("merge2", new MergeVertex(), "up3", "cnn2")

				.addLayer("upcnn3", new ConvolutionLayer.Builder(2, 3).stride(1, 1).nOut(512)
						.activation(new ActivationLReLU(leakyReLUVaue)).convolutionMode(ConvolutionMode.Same).build(),
						"merge2")
				.addLayer("upcnn4",
						new ConvolutionLayer.Builder(2, 2).stride(1, 1).nOut(256)
								.activation(new ActivationLReLU(leakyReLUVaue)).build(),
						"upcnn3")

				//

				.addLayer("cnn9",
						new ConvolutionLayer.Builder(1, 1).nOut(nBoxes * (5 + numClasses)).weightInit(WeightInit.XAVIER)
								.stride(1, 1).weightInit(WeightInit.RELU).activation(Activation.IDENTITY).build(),
						"upcnn4")
				.addLayer("outputs",
						new Yolo2OutputLayer.Builder().lambbaNoObj(lambdaNoObj).lambdaCoord(lambdaCoord)
								.boundingBoxPriors(priors).build(),
						"cnn9")
				.setOutputs("outputs").backprop(true).pretrain(false);

		ComputationGraph neuralNetwork = new ComputationGraph(graphBuilder.build());

		// initialize the network
		neuralNetwork.init();
		return neuralNetwork;
	}

	public static void resBlock(GraphBuilder graphBuilder, String input, int blockNum,
			BaseActivationFunction activation, int... filters) {

		/**
		 * downsample
		 */
		graphBuilder.addLayer("downSamplecnnBlock" + blockNum, new ConvolutionLayer.Builder(3, 3).stride(2, 2)
				.nOut(filters[0]).activation(activation).padding(1, 1).build(), input);
		graphBuilder.addLayer(
				"cnn1Block" + blockNum, new ConvolutionLayer.Builder(1, 1).stride(1, 1).nOut(filters[1])
						.activation(activation).convolutionMode(ConvolutionMode.Same).build(),
				"downSamplecnnBlock" + blockNum);
		/**
		 * next layer does not get activation, its input into shortcut
		 */
		graphBuilder.addLayer("cnn2Block" + blockNum, new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(filters[2])
				.convolutionMode(ConvolutionMode.Same).build(), "cnn1Block" + blockNum);
		graphBuilder.addVertex("shortcut" + blockNum, new ElementWiseVertex(Op.Add), "downSamplecnnBlock" + blockNum,
				"cnn2Block" + blockNum);
		graphBuilder.addLayer("activationBlock" + blockNum,
				new ActivationLayer.Builder().activation(activation).build(), "shortcut" + blockNum);

	}

	public static void resBlocksSmallSample(GraphBuilder graphBuilder, String input, int blockNum, int... filters) {

		/**
		 * downsample
		 */
		graphBuilder.addLayer("cnn1Block" + blockNum, new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(filters[0])
				.activation(new ActivationLReLU()).padding(1, 0).build(), input);
		graphBuilder.addLayer(
				"cnn2Block" + blockNum, new ConvolutionLayer.Builder(1, 1).stride(1, 1).nOut(filters[1])
						.activation(new ActivationReLU()).convolutionMode(ConvolutionMode.Same).build(),
				"cnn1Block" + blockNum);
		/**
		 * next layer does not get activation, its input into shortcut
		 */
		graphBuilder.addLayer("cnn3Block" + blockNum, new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(filters[2])
				.convolutionMode(ConvolutionMode.Same).build(), "cnn2Block" + blockNum);
		graphBuilder.addVertex("shortcut" + blockNum, new ElementWiseVertex(Op.Add), "cnn1Block" + blockNum,
				"cnn3Block" + blockNum);
		graphBuilder.addLayer("activationBlock" + blockNum,
				new ActivationLayer.Builder().activation(new ActivationReLU()).build(), "shortcut" + blockNum);

	}

	public static void shortBlock(GraphBuilder graphBuilder, String input, int blockNum, int... filters) {
		graphBuilder.addLayer("cnn1ShortBlock" + blockNum, new ConvolutionLayer.Builder(1, 1).stride(1, 1)
				.nOut(filters[0]).activation(new ActivationReLU()).convolutionMode(ConvolutionMode.Same).build(),
				input);
		/**
		 * next layer does not get activation, its input into shortcut
		 */
		graphBuilder.addLayer("cnn2ShortBlock" + blockNum, new ConvolutionLayer.Builder(3, 3).stride(1, 1)
				.nOut(filters[1]).convolutionMode(ConvolutionMode.Same).build(), "cnn1ShortBlock" + blockNum);
		graphBuilder.addVertex("shortcutShortBlock" + blockNum, new ElementWiseVertex(Op.Add),
				"cnn2ShortBlock" + blockNum, input);
		graphBuilder.addLayer("activationBlock" + blockNum,
				new ActivationLayer.Builder().activation(new ActivationReLU()).build(),
				"shortcutShortBlock" + blockNum);

	}

	public static void main(String[] args) {
		/**
		 * just to see architecture
		 */
		double[][] priorBoxes = FaultUtils.allPriors;
		// Models.CLASModel(216, 112, 3);

		// Models.test(216, 112, 3);
		// Models.KunkelPeterModel(6, 112, 3, 2);
		ComputationGraph test = Models.DriftChamber(12, 112, 1, 2, priorBoxes);
		System.out.println(test.summary(InputType.convolutional(12, 112, 1)));

	}

}
