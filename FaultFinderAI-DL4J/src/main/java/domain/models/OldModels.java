package domain.models;

import java.util.ArrayList;
import java.util.stream.Collectors;
import java.util.stream.Stream;

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
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.Upsampling2D;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.activations.impl.ActivationLReLU;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.impl.LossNegativeLogLikelihood;

import clasDC.faults.FaultNames;
import clasDC.objects.CLASObject;
import clasDC.objects.DriftChamber;
import utils.FaultUtils;

/**
 * This class is used to retrieve all the different models that are available.
 */
public class OldModels {
	public static ComputationGraph KunkelPeterModel(int height, int width, int numChannels, int numLabels) {

		GraphBuilder graphBuilder = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(new Adam()).graphBuilder()
				.addInputs("input").setInputTypes(InputType.convolutional(height, width, numChannels))
				.addLayer("cnn1",
						new ConvolutionLayer.Builder(2, 3).nIn(numChannels).stride(1, 1).nOut(40)
								.activation(new ActivationReLU()).build(),
						"input")
				.addLayer("cnn2",
						new ConvolutionLayer.Builder(2, 2).nIn(40).stride(1, 1).nOut(30)
								.activation(new ActivationReLU()).build(),
						"cnn1")
				.addLayer("pool1",
						new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2)
								.build(),
						"cnn2")
				.addLayer("cnn3",
						new ConvolutionLayer.Builder(2, 2).nIn(30).stride(1, 1).nOut(20)
								.activation(new ActivationReLU()).build(),
						"pool1")
				.addLayer("dense1", new DenseLayer.Builder().activation(new ActivationReLU()).nOut(100).build(), "cnn3")
				.addLayer("out", new OutputLayer.Builder(new LossNegativeLogLikelihood()).nIn(100).nOut(numLabels)
						.activation(new ActivationSoftmax()).build(), "dense1")
				.setOutputs("out");

		ComputationGraph neuralNetwork = new ComputationGraph(graphBuilder.backprop(true).pretrain(false).build());

		// initialize the network
		neuralNetwork.init();
		System.out.println(neuralNetwork.summary(InputType.convolutional(height, width, numChannels)));

		return neuralNetwork;
	}

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
				.setInputTypes(InputType.convolutional(height, width, numChannels));
		resBlock(graphBuilder, "input", 1, 64, 128, 64);
		resBlock(graphBuilder, ("activationBlock" + 1), 2, 128, 256, 128);
		/**
		 * set up yolo
		 */

		graphBuilder
				/**
				 * I need this to make the width an odd number
				 */
				.addLayer("cnn3", new ConvolutionLayer.Builder(2, 3).stride(1, 1).nOut(128)
						.activation(new ActivationLReLU(leakyReLUVaue)).convolutionMode(ConvolutionMode.Same).build(),
						"activationBlock" + 2)
				//
				.addLayer("cnn4", new ConvolutionLayer.Builder(2, 2).stride(1, 1).nOut(128)
						.activation(new ActivationLReLU(leakyReLUVaue)).convolutionMode(ConvolutionMode.Same).build(),
						"cnn3")

				.addLayer("up3", new Upsampling2D.Builder().size(new int[] { 2, 2 }).build(), "cnn4")

				.addVertex("merge2", new MergeVertex(), "up3", "activationBlock" + 2)
//				.addLayer("oddcnn",
//						new ConvolutionLayer.Builder(2, 2).stride(1, 1).nOut(128)
//								.activation(new ActivationLReLU(leakyReLUVaue)).build(),
//						"activationBlock" + 2)
				.addLayer("upcnn3", new ConvolutionLayer.Builder(2, 3).stride(1, 1).nOut(512)
						.activation(new ActivationLReLU(leakyReLUVaue)).convolutionMode(ConvolutionMode.Same).build(),
						"merge2")
				.addLayer("upcnn4",
						new ConvolutionLayer.Builder(2, 2).stride(1, 1).nOut(256)
								.activation(new ActivationLReLU(leakyReLUVaue)).build(),
						"upcnn3")
				.addLayer("cnnBeforeYolo",
						new ConvolutionLayer.Builder(1, 1).nOut(nBoxes * (5 + numClasses)).weightInit(WeightInit.XAVIER)
								.stride(1, 1).convolutionMode(ConvolutionMode.Same).activation(new ActivationIdentity())
								.build(),
						"upcnn4")
				.addLayer("outputs",
						new Yolo2OutputLayer.Builder().lambbaNoObj(lambdaNoObj).lambdaCoord(lambdaCoord)
								.boundingBoxPriors(priors).build(),
						"cnnBeforeYolo")
				.setOutputs("outputs").backprop(true).pretrain(false);

		ComputationGraph neuralNetwork = new ComputationGraph(graphBuilder.backprop(true).pretrain(false).build());

		// initialize the network
		neuralNetwork.init();
		// System.out.println(neuralNetwork.summary(InputType.convolutional(height,
		// width, numChannels)));

		return neuralNetwork;

	}

	public static ComputationGraph KunkelPetersYolo(int height, int width, int numChannels, int numClasses,
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
				.addLayer("cnn1",
						new ConvolutionLayer.Builder(2, 3).stride(1, 1).nOut(40)
								.activation(new ActivationLReLU(leakyReLUVaue)).build(),
						"input")
				.addLayer("cnn2",
						new ConvolutionLayer.Builder(2, 2).stride(1, 1).nOut(30)
								.activation(new ActivationLReLU(leakyReLUVaue)).build(),
						"cnn1")
				.addLayer("pool1",
						new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2)
								.build(),
						"cnn2")
				.addLayer("cnn3",
						new ConvolutionLayer.Builder(2, 2).stride(1, 1).nOut(20)
								.activation(new ActivationLReLU(leakyReLUVaue)).build(),
						"pool1")
				.addLayer("cnn4",
						new ConvolutionLayer.Builder(2, 1).stride(1, 1).nOut(20)
								.activation(new ActivationLReLU(leakyReLUVaue)).build(),
						"cnn3")
				.addLayer("cnn5",
						new ConvolutionLayer.Builder(1, 1).nOut(nBoxes * (5 + numClasses)).weightInit(WeightInit.XAVIER)
								.stride(1, 1).weightInit(WeightInit.RELU).activation(Activation.IDENTITY).build(),
						"cnn4")
				.addLayer("outputs",
						new Yolo2OutputLayer.Builder().lambbaNoObj(lambdaNoObj).lambdaCoord(lambdaCoord)
								.boundingBoxPriors(priors).build(),
						"cnn5")
				.setOutputs("outputs").backprop(true).pretrain(false);

		ComputationGraph neuralNetwork = new ComputationGraph(graphBuilder.build());

		// initialize the network
		neuralNetwork.init();
		// System.out.println(neuralNetwork.summary(InputType.convolutional(height,
		// width, numChannels)));

		return neuralNetwork;
	}

	public static ComputationGraph KunkelPetersUYolo4SL(int height, int width, int numChannels) {

		int numClasses = 11;
		double[][] priorBoxes = FaultUtils.getPriors(new double[][] { { height / 5, width / 111 } });// FaultUtils.allPriors

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

				.addLayer("beforeOutput",
						new ConvolutionLayer.Builder(1, 1).nOut(nBoxes * (5 + numClasses)).weightInit(WeightInit.XAVIER)
								.stride(1, 1).weightInit(WeightInit.RELU).activation(Activation.IDENTITY).build(),
						"upcnn4")
				.addLayer("outputs",
						new Yolo2OutputLayer.Builder().lambbaNoObj(lambdaNoObj).lambdaCoord(lambdaCoord)
								.boundingBoxPriors(priors).build(),
						"beforeOutput")
				.setOutputs("outputs").backprop(true).pretrain(false);

		ComputationGraph neuralNetwork = new ComputationGraph(graphBuilder.build());

		// initialize the network
		neuralNetwork.init();
		System.out.println(neuralNetwork.summary(InputType.convolutional(height, width, numChannels)));

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

	public static ComputationGraph KunkelPetersUYolo(int height, int width, int numChannels) {

		int numClasses = 11;
		double[][] priorBoxes = FaultUtils.allPriors;
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
				.addLayer("pool2",
						new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2)
								.build(),
						"cnn4")
				//

				.addLayer("cnn5", new ConvolutionLayer.Builder(1, 3).stride(1, 1).nOut(256)
						.activation(new ActivationLReLU(leakyReLUVaue)).convolutionMode(ConvolutionMode.Same).build(),
						"pool2")
				.addLayer("cnn6", new ConvolutionLayer.Builder(1, 2).stride(1, 1).nOut(256)
						.activation(new ActivationLReLU(leakyReLUVaue)).convolutionMode(ConvolutionMode.Same).build(),
						"cnn5")
				.addLayer("pool3",
						new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2)
								.build(),
						"cnn6")

				.addLayer("cnn7", new ConvolutionLayer.Builder(1, 3).stride(1, 1).nOut(512)
						.activation(new ActivationLReLU(leakyReLUVaue)).convolutionMode(ConvolutionMode.Same).build(),
						"pool3")
				.addLayer("cnn8", new ConvolutionLayer.Builder(1, 2).stride(1, 1).nOut(512)
						.activation(new ActivationLReLU(leakyReLUVaue)).convolutionMode(ConvolutionMode.Same).build(),
						"cnn7")
				//
				// .addLayer("up1", new Upsampling2D.Builder().size(new int[] {
				// 2, 2 }).build(),
				// "cnn8")
				.addLayer("oddcnn",
						new ConvolutionLayer.Builder(1, 1).stride(1, 1).nOut(1024)
								.activation(new ActivationLReLU(leakyReLUVaue)).padding(1, 0).build(),
						"cnn8")
				.addLayer("up1", new Upsampling2D.Builder().size(new int[] { 1, 2 }).build(), "oddcnn")
				// .addLayer("oddcnn1", new ConvolutionLayer.Builder(2,
				// 1).stride(1,
				// 1).nOut(512)
				// .activation(new ActivationLReLU(leakyReLUVaue)).build(),
				// "up1")
				//
				//

				.addVertex("merge1", new MergeVertex(), "up1", "cnn6")
				// again KPModel
				.addLayer("upcnn1", new ConvolutionLayer.Builder(2, 3).stride(1, 1).nOut(512)
						.activation(new ActivationLReLU(leakyReLUVaue)).convolutionMode(ConvolutionMode.Same).build(),
						"merge1")
				.addLayer("upcnn2", new ConvolutionLayer.Builder(2, 2).stride(1, 1).nOut(512)
						.activation(new ActivationLReLU(leakyReLUVaue)).convolutionMode(ConvolutionMode.Same).build(),
						"upcnn1")
				.addLayer("up2", new Upsampling2D.Builder().size(new int[] { 2, 2 }).build(), "upcnn2")

				.addVertex("merge2", new MergeVertex(), "up2", "cnn4")

				.addLayer("upcnn3", new ConvolutionLayer.Builder(2, 3).stride(1, 1).nOut(512)
						.activation(new ActivationLReLU(leakyReLUVaue)).convolutionMode(ConvolutionMode.Same).build(),
						"merge2")
				.addLayer("upcnn4", new ConvolutionLayer.Builder(2, 2).stride(1, 1).nOut(256)
						.activation(new ActivationLReLU(leakyReLUVaue)).convolutionMode(ConvolutionMode.Same).build(),
						"upcnn3")

				.addLayer("up3", new Upsampling2D.Builder().size(new int[] { 2, 2 }).build(), "upcnn4")

				.addVertex("merge3", new MergeVertex(), "up3", "cnn2")

				.addLayer("upcnn5", new ConvolutionLayer.Builder(2, 3).stride(1, 1).nOut(128)
						.activation(new ActivationLReLU(leakyReLUVaue)).convolutionMode(ConvolutionMode.Same).build(),
						"merge3")
				.addLayer("upcnn6", new ConvolutionLayer.Builder(2, 2).stride(1, 1).nOut(64)
						.activation(new ActivationLReLU(leakyReLUVaue)).convolutionMode(ConvolutionMode.Same).build(),
						"upcnn5")

				//

				.addLayer("cnn9",
						new ConvolutionLayer.Builder(1, 1).nOut(nBoxes * (5 + numClasses)).weightInit(WeightInit.XAVIER)
								.stride(1, 1).weightInit(WeightInit.RELU).activation(Activation.IDENTITY).build(),
						"upcnn6")
				.addLayer("outputs",
						new Yolo2OutputLayer.Builder().lambbaNoObj(lambdaNoObj).lambdaCoord(lambdaCoord)
								.boundingBoxPriors(priors).build(),
						"cnn9")
				.setOutputs("outputs").backprop(true).pretrain(false);

		ComputationGraph neuralNetwork = new ComputationGraph(graphBuilder.build());

		// initialize the network
		neuralNetwork.init();
		System.out.println(neuralNetwork.summary(InputType.convolutional(height, width, numChannels)));

		return neuralNetwork;
	}

	public static ComputationGraph KPYolo3(int height, int width, int numChannels) {

		int numClasses = 14;
		double[][] priorBoxes = FaultUtils.allPriors;
		int nBoxes = priorBoxes.length;

		INDArray priors = Nd4j.create(priorBoxes);
		double lambdaNoObj = 0.5;
		double lambdaCoord = 1.0;

		GraphBuilder graphBuilder = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER).l2(0.0005)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(new Adam()).graphBuilder()
				.addInputs("input").setInputTypes(InputType.convolutional(height, width, numChannels))
				.addLayer("cnn1", new ConvolutionLayer.Builder(3, 3).nIn(numChannels).stride(1, 1)
						.convolutionMode(ConvolutionMode.Same).nOut(32).activation(new ActivationReLU()).build(),
						"input")
				.addLayer("cnn2",
						new ConvolutionLayer.Builder(3, 3).nIn(64).stride(1, 1).convolutionMode(ConvolutionMode.Same)
								.nOut(128).activation(new ActivationReLU()).build(),
						"cnn1")
				.addLayer("cnn3",
						new ConvolutionLayer.Builder(2, 2).nIn(128).stride(1, 1).convolutionMode(ConvolutionMode.Same)
								.nOut(256).build(),
						"cnn2")
				// .addVertex("mergedLayer", new MergeVertex(), "cnn1", "cnn3")
				.addLayer("activateMerged", new ActivationLayer.Builder().activation(new ActivationReLU()).build(),
						"input", "cnn3")
				.addLayer("cnn4",
						new ConvolutionLayer.Builder(2, 2).nIn(256 + 1).stride(1, 1).nOut(128)
								.activation(new ActivationReLU()).build(),
						"activateMerged")
				.addLayer("cnn5",
						new ConvolutionLayer.Builder(1, 1).nIn(128).nOut(nBoxes * (5 + numClasses))
								.weightInit(WeightInit.XAVIER).stride(1, 1).weightInit(WeightInit.RELU)
								.activation(Activation.IDENTITY).build(),
						"cnn4")
				.addLayer("outputs",
						new Yolo2OutputLayer.Builder().lambbaNoObj(lambdaNoObj).lambdaCoord(lambdaCoord)
								.boundingBoxPriors(priors).build(),
						"cnn5")
				.setOutputs("outputs").backprop(true).pretrain(false);

		ComputationGraph neuralNetwork = new ComputationGraph(graphBuilder.build());

		// initialize the network
		neuralNetwork.init();
		System.out.println(neuralNetwork.summary(InputType.convolutional(height, width, numChannels)));

		return neuralNetwork;
	}

	public static ComputationGraph singleSuperlayerModel(int height, int width, int numChannels) {
		/**
		 * This model is inspired by the KunkelPeters model. See ModelFactory.deeperCNN
		 * Its use is not for object detection, but rather classification of faults.
		 * Does not give fault location, but trains and works on data extremely well.
		 * Superlayer is too small of an 'image' for object detection
		 */
		int numClasses = 14;
		double[][] priorBoxes = FaultUtils.allPriors;
		int nBoxes = priorBoxes.length;

		INDArray priors = Nd4j.create(priorBoxes);
		double lambdaNoObj = 0.5;
		double lambdaCoord = 1.0;
		GraphBuilder graphBuilder = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(new Adam()).graphBuilder()
				.addInputs("input").setInputTypes(InputType.convolutional(height, width, numChannels))
				.addLayer("cnn1", new ConvolutionLayer.Builder(2, 3).nIn(numChannels).stride(1, 1)
						.convolutionMode(ConvolutionMode.Same).nOut(32).activation(new ActivationReLU()).build(),
						"input")
				.addLayer("cnn2",
						new ConvolutionLayer.Builder(2, 2).nIn(32).stride(1, 1).convolutionMode(ConvolutionMode.Same)
								.nOut(64).activation(new ActivationReLU()).build(),
						"cnn1")
				.addLayer("pool1",
						new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2)
								.build(),
						"cnn2")
				.addLayer("cnn3",
						new ConvolutionLayer.Builder(2, 2).nIn(64).stride(1, 1).convolutionMode(ConvolutionMode.Same)
								.nOut(128).activation(new ActivationReLU()).build(),
						"pool1")
				.addLayer("insert1",
						new ConvolutionLayer.Builder(1, 2).nIn(128).stride(1, 1).convolutionMode(ConvolutionMode.Same)
								.nOut(256).activation(new ActivationReLU()).build(),
						"cnn3")
				.addLayer("pool2",
						new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(1, 2).stride(1, 2)
								.build(),
						"insert1")
				.addLayer("insert2",
						new ConvolutionLayer.Builder(2, 2).nIn(256).stride(1, 1).convolutionMode(ConvolutionMode.Same)
								.nOut(512).activation(new ActivationReLU()).build(),
						"pool2")
				.addLayer("cnn4",
						new ConvolutionLayer.Builder(1, 1).nIn(512).nOut(nBoxes * (5 + numClasses))
								.weightInit(WeightInit.XAVIER).stride(1, 1).convolutionMode(ConvolutionMode.Same)
								.weightInit(WeightInit.RELU).activation(Activation.IDENTITY).build(),
						"insert2")
				.addLayer("outputs",
						new Yolo2OutputLayer.Builder().lambbaNoObj(lambdaNoObj).lambdaCoord(lambdaCoord)
								.boundingBoxPriors(priors).build(),
						"cnn4")
				.setOutputs("outputs").backprop(true).pretrain(false);

		ComputationGraph neuralNetwork = new ComputationGraph(graphBuilder.build());

		// initialize the network
		neuralNetwork.init();
		System.out.println(neuralNetwork.summary(InputType.convolutional(height, width, numChannels)));

		return neuralNetwork;
	}

	public static void resBlockDC(GraphBuilder graphBuilder, String input, double leakyReLUVaue, int blockNum,
			int... filters) {

		/**
		 * downsample
		 */

		graphBuilder.addLayer("cnn1Block" + blockNum, new ConvolutionLayer.Builder(2, 3).stride(1, 1).nOut(filters[0])
				.activation(new ActivationLReLU(leakyReLUVaue)).build(), input);
		graphBuilder.addLayer("cnn2Block" + blockNum,
				new ConvolutionLayer.Builder(1, 1).stride(1, 1).nOut(filters[1])
						.activation(new ActivationLReLU(leakyReLUVaue)).convolutionMode(ConvolutionMode.Same).build(),
				"cnn1Block" + blockNum);
		/**
		 * next layer does not get activation, its input into shortcut
		 */
		graphBuilder.addLayer("cnn3Block" + blockNum, new ConvolutionLayer.Builder(2, 2).stride(1, 1).nOut(filters[2])
				.convolutionMode(ConvolutionMode.Same).build(), "cnn2Block" + blockNum);
		graphBuilder.addVertex("shortcut" + blockNum, new ElementWiseVertex(Op.Add), "cnn1Block" + blockNum,
				"cnn3Block" + blockNum);
		graphBuilder.addLayer("activationBlock" + blockNum,
				new ActivationLayer.Builder().activation(new ActivationReLU()).build(), "shortcut" + blockNum);

	}

	public static ComputationGraph DCModel(int height, int width, int numChannels) {
		/**
		 *
		 * This is a single driftchamber DC model Inspired by the KunkelPetersModel for
		 * classification. Here down sample will be done by ConvolutionLayers so remove
		 * Subsampling layer
		 *
		 * The final dimensions of this should be height =7 , width = 45 need to scale
		 * priors by these for YOLO2 layer, if YOLO3 layer, do not scale
		 */
		/**
		 * with HotWire, DeadWire and NoFault there are 14 classes, but I suspect that
		 * these wires are not performing correctly. Lets test that theory.
		 */
		int numClasses = 11;
		/**
		 * convert priors to this models scaling
		 */
		double[][] priorBoxes = FaultUtils.getPriors(new double[][] { { height / 11, width / 53 } });// FaultUtils.allPriors
		// ;
		int nBoxes = priorBoxes.length;

		INDArray priors = Nd4j.create(priorBoxes);
		double lambdaNoObj = 0.5;
		double lambdaCoord = 1.0;
		// .l1(1e-7)
		// .l2(0.005)
		double learningRate = 1e-4;
		// goes on AdamUpdater
		// .learningRate(learningRate)
		double l2Rate = 0.0001;
		int seed = 123;
		double leakyReLUVaue = 0.1;
		GraphBuilder graphBuilder = new NeuralNetConfiguration.Builder().seed(seed)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.gradientNormalization(GradientNormalization.RenormalizeL2PerLayer).gradientNormalizationThreshold(1.0)
				.weightInit(WeightInit.XAVIER).updater(new Adam.Builder().learningRate(learningRate).build()).l2(l2Rate)
				.activation(Activation.IDENTITY).graphBuilder().addInputs("input")
				.setInputTypes(InputType.convolutional(height, width, numChannels))
				.addLayer("startercnn", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(32)
						.activation(new ActivationLReLU(leakyReLUVaue)).convolutionMode(ConvolutionMode.Same).build(),
						"input")
				/**
				 * rest is inspired by KunkelPetersModel Except the firstlayer will down sample
				 */
				.addLayer("cnn1", new ConvolutionLayer.Builder(2, 3).stride(1, 2).nOut(64)
						.activation(new ActivationLReLU(leakyReLUVaue)).convolutionMode(ConvolutionMode.Same).build(),
						"startercnn")
				.addLayer("cnn2", new ConvolutionLayer.Builder(2, 2).stride(1, 1).nOut(32)
						.activation(new ActivationLReLU(leakyReLUVaue)).convolutionMode(ConvolutionMode.Same).build(),
						"cnn1")
				.addLayer("cnn3",
						new ConvolutionLayer.Builder(2, 2).stride(1, 1).nOut(64).convolutionMode(ConvolutionMode.Same)
								.build(),
						"cnn2")
				.addVertex("shortcut", new ElementWiseVertex(Op.Add), "cnn1", "cnn3").addLayer("activationBlock",
						new ActivationLayer.Builder().activation(new ActivationReLU()).build(), "shortcut");

		/**
		 * downsample
		 */
		resBlockDC(graphBuilder, "activationBlock", leakyReLUVaue, 1, 64, 128, 64);
		// resBlockDC(graphBuilder, "activationBlock" + 1, leakyReLUVaue, 2,
		// 128, 256,
		// 128);
		// resBlockDC(graphBuilder, "activationBlock" + 2, leakyReLUVaue, 3,
		// 256, 512,
		// 256);
		// resBlockDC(graphBuilder, "activationBlock" + 3, leakyReLUVaue, 4,
		// 512, 1024,
		// 512);
		// resBlockDC(graphBuilder, "activationBlock" + 4, leakyReLUVaue, 5,
		// 1024, 2048,
		// 1024);

		/**
		 * set up yolo
		 */

		graphBuilder
				/**
				 * I need this to make the width an odd number
				 */
				.addLayer("oddcnn",
						new ConvolutionLayer.Builder(1, 2).stride(1, 1).nOut(2048)
								.activation(new ActivationLReLU(leakyReLUVaue)).build(),
						"activationBlock" + 1)
				.addLayer("cnn4",
						new ConvolutionLayer.Builder(1, 1).nOut(nBoxes * (5 + numClasses)).weightInit(WeightInit.XAVIER)
								.stride(1, 1).convolutionMode(ConvolutionMode.Same).activation(new ActivationIdentity())
								.build(),
						"oddcnn")
				.addLayer("outputs",
						new Yolo2OutputLayer.Builder().lambbaNoObj(lambdaNoObj).lambdaCoord(lambdaCoord)
								.boundingBoxPriors(priors).build(),
						"cnn4")
				.setOutputs("outputs").backprop(true).pretrain(false);

		ComputationGraph neuralNetwork = new ComputationGraph(graphBuilder.backprop(true).pretrain(false).build());

		// initialize the network
		neuralNetwork.init();
		System.out.println(neuralNetwork.summary(InputType.convolutional(height, width, numChannels)));

		return neuralNetwork;
	}

	public static ComputationGraph RegionhModel(int height, int width, int numChannels) {

		int numClasses = 14;
		double[][] priorBoxes = FaultUtils.allPriors;
		int nBoxes = priorBoxes.length;

		INDArray priors = Nd4j.create(priorBoxes);
		double lambdaNoObj = 0.5;
		double lambdaCoord = 1.0;
		GraphBuilder graphBuilder = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(new Adam()).graphBuilder()
				.addInputs("input").setInputTypes(InputType.convolutional(height, width, numChannels))
				.addLayer("cnn1", new ConvolutionLayer.Builder(2, 3).nIn(numChannels).stride(1, 1)
						.convolutionMode(ConvolutionMode.Same).nOut(32).activation(new ActivationReLU()).build(),
						"input")
				.addLayer("cnn2",
						new ConvolutionLayer.Builder(2, 2).nIn(32).stride(1, 1).convolutionMode(ConvolutionMode.Same)
								.nOut(64).activation(new ActivationReLU()).build(),
						"cnn1")
				.addLayer("pool1",
						new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2)
								.build(),
						"cnn2")
				.addLayer("cnn3",
						new ConvolutionLayer.Builder(2, 2).nIn(64).stride(1, 1).convolutionMode(ConvolutionMode.Same)
								.nOut(128).activation(new ActivationReLU()).build(),
						"pool1")
				.addLayer("insert1",
						new ConvolutionLayer.Builder(1, 2).nIn(128).stride(1, 1).convolutionMode(ConvolutionMode.Same)
								.nOut(256).activation(new ActivationReLU()).build(),
						"cnn3")
				.addLayer("pool2",
						new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(1, 2).stride(1, 2)
								.build(),
						"insert1")
				.addLayer("insert2",
						new ConvolutionLayer.Builder(2, 2).nIn(256).stride(1, 1).convolutionMode(ConvolutionMode.Same)
								.nOut(512).activation(new ActivationReLU()).build(),
						"pool2")
				.addLayer("cnn4",
						new ConvolutionLayer.Builder(1, 1).nIn(512).nOut(nBoxes * (5 + numClasses))
								.weightInit(WeightInit.XAVIER).stride(1, 1).convolutionMode(ConvolutionMode.Same)
								.weightInit(WeightInit.RELU).activation(Activation.IDENTITY).build(),
						"insert2")
				.addLayer("outputs",
						new Yolo2OutputLayer.Builder().lambbaNoObj(lambdaNoObj).lambdaCoord(lambdaCoord)
								.boundingBoxPriors(priors).build(),
						"cnn4")
				.setOutputs("outputs").backprop(true).pretrain(false);

		ComputationGraph neuralNetwork = new ComputationGraph(graphBuilder.build());

		// initialize the network
		neuralNetwork.init();
		System.out.println(neuralNetwork.summary(InputType.convolutional(height, width, numChannels)));

		return neuralNetwork;
	}

	public static ComputationGraph CLASModel(int height, int width, int numChannels) {

		int numClasses = 14;
		double[][] priorBoxes = FaultUtils.allPriors;
		int nBoxes = priorBoxes.length;

		INDArray priors = Nd4j.create(priorBoxes);
		double lambdaNoObj = 0.5;
		double lambdaCoord = 1.0;
		GraphBuilder graphBuilder = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(new Adam()).graphBuilder()
				.addInputs("input").setInputTypes(InputType.convolutional(height, width, numChannels))
				.addLayer("cnn1",
						new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(32).activation(new ActivationLReLU())
								.convolutionMode(ConvolutionMode.Same).build(),
						"input")
				/**
				 * downsample
				 */
				.addLayer("cnn2",
						new ConvolutionLayer.Builder(3, 3).stride(2, 2).nOut(64).activation(new ActivationLReLU())
								.padding(1, 1).build(),
						"cnn1")
				.addLayer("cnn3",
						new ConvolutionLayer.Builder(1, 1).stride(1, 1).nOut(32).activation(new ActivationReLU())
								.convolutionMode(ConvolutionMode.Same).build(),
						"cnn2")
				.addLayer("cnn4",
						new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(64).activation(new ActivationReLU())
								.convolutionMode(ConvolutionMode.Same).build(),
						"cnn3")
				.addVertex("shortcut1", new ElementWiseVertex(Op.Add), "cnn2", "cnn4")
				/**
				 * downsample
				 */
				.addLayer("cnn5",
						new ConvolutionLayer.Builder(3, 3).stride(2, 2).nOut(128).activation(new ActivationLReLU())
								.padding(1, 1).build(),
						"shortcut1")
				.addLayer("cnn6",
						new ConvolutionLayer.Builder(1, 1).stride(1, 1).nOut(64).activation(new ActivationReLU())
								.convolutionMode(ConvolutionMode.Same).build(),
						"cnn5")

				.addLayer("cnn7",
						new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(128).activation(new ActivationReLU())
								.convolutionMode(ConvolutionMode.Same).build(),
						"cnn6")
				// .addVertex("shortcut1", new ElementWiseVertex(Op.Add),
				// "cnn2", "cnn4")

				// .addLayer("cnn4",
				// new ConvolutionLayer.Builder(1, 1).nIn(512).nOut(nBoxes * (5
				// + numClasses))
				// .weightInit(WeightInit.XAVIER).stride(1,
				// 1).convolutionMode(ConvolutionMode.Same)
				// .weightInit(WeightInit.RELU).activation(Activation.IDENTITY).build(),
				// "insert2")
				// .addLayer("outputs",
				// new
				// Yolo2OutputLayer.Builder().lambbaNoObj(lambdaNoObj).lambdaCoord(lambdaCoord)
				// .boundingBoxPriors(priors).build(),
				// "cnn4")
				.setOutputs("cnn7").backprop(true).pretrain(false);

		ComputationGraph neuralNetwork = new ComputationGraph(graphBuilder.build());

		// initialize the network
		neuralNetwork.init();
		System.out.println(neuralNetwork.summary(InputType.convolutional(height, width, numChannels)));

		return neuralNetwork;
	}

	public static void resBlock(GraphBuilder graphBuilder, String input, int blockNum, int... filters) {

		/**
		 * downsample
		 */
		graphBuilder.addLayer("cnn1Block" + blockNum, new ConvolutionLayer.Builder(3, 3).stride(2, 2).nOut(filters[0])
				.activation(new ActivationLReLU()).padding(1, 1).build(), input);
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

	public static void resBlocksSmallSample(GraphBuilder graphBuilder, String input, int blockNum, int... filters) {

		/**
		 * downsample
		 */
		graphBuilder.addLayer("cnn1Block" + blockNum, new ConvolutionLayer.Builder(3, 3).stride(2, 1).nOut(filters[0])
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

	public static ComputationGraph test(int height, int width, int numChannels) {

		int numClasses = 14;
		double[][] priorBoxes = FaultUtils.allPriors;
		int nBoxes = priorBoxes.length;

		INDArray priors = Nd4j.create(priorBoxes);
		double lambdaNoObj = 0.5;
		double lambdaCoord = 1.0;
		GraphBuilder graphBuilder = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(new Adam()).graphBuilder();

		graphBuilder.addInputs("input").setInputTypes(InputType.convolutional(height, width, numChannels));
		graphBuilder.addLayer("cnn1", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(32)
				.activation(new ActivationLReLU()).convolutionMode(ConvolutionMode.Same).build(), "input");
		resBlocksSmallSample(graphBuilder, "cnn1", 1, 64, 32, 64);
		resBlocksSmallSample(graphBuilder, "shortcut" + 1, 2, 128, 64, 128);
		shortBlock(graphBuilder, "shortcut" + 2, 2, 64, 128);
		resBlocksSmallSample(graphBuilder, "shortcutShortBlock" + 2, 3, 256, 128, 256);
		shortBlock(graphBuilder, "shortcut" + 3, 3, 128, 256);
		shortBlock(graphBuilder, "shortcutShortBlock" + 3, 31, 128, 256);
		shortBlock(graphBuilder, "shortcutShortBlock" + 31, 32, 128, 256);
		shortBlock(graphBuilder, "shortcutShortBlock" + 32, 33, 128, 256);
		shortBlock(graphBuilder, "shortcutShortBlock" + 33, 34, 128, 256);
		shortBlock(graphBuilder, "shortcutShortBlock" + 34, 35, 128, 256);
		shortBlock(graphBuilder, "shortcutShortBlock" + 35, 36, 128, 256);

		resBlock(graphBuilder, "shortcutShortBlock" + 36, 4, 512, 256, 512);
		shortBlock(graphBuilder, "shortcut" + 4, 4, 256, 512);
		shortBlock(graphBuilder, "shortcutShortBlock" + 4, 41, 256, 512);
		shortBlock(graphBuilder, "shortcutShortBlock" + 41, 42, 256, 512);
		shortBlock(graphBuilder, "shortcutShortBlock" + 42, 43, 256, 512);
		shortBlock(graphBuilder, "shortcutShortBlock" + 43, 44, 256, 512);
		shortBlock(graphBuilder, "shortcutShortBlock" + 44, 45, 256, 512);
		shortBlock(graphBuilder, "shortcutShortBlock" + 45, 46, 256, 512);

		// at line 459, i.e. need to add downsampling
		/**
		 * https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
		 * 
		 * some refs
		 * https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-2/
		 * https://www.kdnuggets.com/2018/05/implement-yolo-v3-object-detector-pytorch-part-1.html
		 * https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b
		 * https://github.com/pjreddie/darknet/issues/568
		 * 
		 * https://deeplearning4j.org/docs/latest/deeplearning4j-nn-computationgraph
		 */
		graphBuilder.setOutputs("shortcutShortBlock" + 46).backprop(true).pretrain(false);
		// graphBuilder.setOutputs("shortcut" +
		// 4).backprop(true).pretrain(false);

		ComputationGraph neuralNetwork = new ComputationGraph(graphBuilder.build());

		// initialize the network
		neuralNetwork.init();
		System.out.println(neuralNetwork.summary(InputType.convolutional(height, width, numChannels)));

		return neuralNetwork;
	}

	public static void main(String[] args) {
		CLASObject object = DriftChamber.builder().region(1).nchannels(1).maxFaults(10)
				.desiredFaults(Stream.of(FaultNames.CONNECTOR_TREE, FaultNames.CONNECTOR_TREE, FaultNames.CHANNEL_THREE,
						FaultNames.PIN_SMALL).collect(Collectors.toCollection(ArrayList::new)))
				.singleFaultGen(true).build();
		CLASModelFactory mFactory = new CLASModelFactory(object);
		System.out.println(mFactory.getGridWidth() + "  " + mFactory.getGridHeight());
		mFactory.getComputationGraph();
	}

}
