package domain.models;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex.Op;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.Upsampling2D;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
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
	public static ComputationGraph KunkelPetersYolo(int height, int width, int numChannels) {

		int numClasses = 14;
		double[][] priorBoxes = FaultUtils.allPriors;
		int nBoxes = priorBoxes.length;

		INDArray priors = Nd4j.create(priorBoxes);
		double lambdaNoObj = 0.5;
		double lambdaCoord = 1.0;

		GraphBuilder graphBuilder = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER).l2(0.005)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(new Adam()).graphBuilder()
				.addInputs("input").setInputTypes(InputType.convolutional(height, width, numChannels))
				.addLayer("cnn1", new ConvolutionLayer.Builder(3, 3).nIn(numChannels).stride(1, 1)
						.convolutionMode(ConvolutionMode.Same).nOut(64).activation(new ActivationReLU()).build(),
						"input")
				.addLayer("cnn2",
						new ConvolutionLayer.Builder(3, 3).nIn(64).stride(1, 1).convolutionMode(ConvolutionMode.Same)
								.nOut(128).activation(new ActivationReLU()).build(),
						"cnn1")
				.addLayer("cnn3",
						new ConvolutionLayer.Builder(2, 2).nIn(128).stride(1, 1).convolutionMode(ConvolutionMode.Same)
								.nOut(256).activation(new ActivationReLU()).build(),
						"cnn2")
				.addLayer("activateMerged", new ActivationLayer.Builder().activation(new ActivationReLU()).build(),
						"mergedLayer")
				.addLayer("cnn4",
						new ConvolutionLayer.Builder(2, 2).nIn(256).stride(1, 1).nOut(128)
								.activation(new ActivationReLU()).build(),
						"cnn3")
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

	public static ComputationGraph DCModel(int height, int width, int numChannels) {

		int numClasses = 14;
		double[][] priorBoxes = FaultUtils.allPriors;
		int nBoxes = priorBoxes.length;

		INDArray priors = Nd4j.create(priorBoxes);
		double lambdaNoObj = 0.5;
		double lambdaCoord = 1.0;
		// .l1(1e-7)
		// .l2(0.005)
		GraphBuilder graphBuilder = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(new Adam()).graphBuilder()
				.addInputs("input").setInputTypes(InputType.convolutional(height, width, numChannels))
				.addLayer("cnn1", new ConvolutionLayer.Builder(2, 3).nIn(numChannels).stride(1, 1)
						.convolutionMode(ConvolutionMode.Same).nOut(32).activation(new ActivationReLU()).build(),
						"input")

				.addLayer("upsample", new Upsampling2D.Builder().size(4).build(), "cnn1")

				.addLayer("cnn2",
						new ConvolutionLayer.Builder(3, 3).nIn(32).stride(1, 1).convolutionMode(ConvolutionMode.Same)
								.nOut(64).activation(new ActivationReLU()).build(),
						"upsample")
				.addLayer("pool1",
						new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2)
								.build(),
						"cnn2")
				.addLayer("cnn3",
						new ConvolutionLayer.Builder(2, 2).nIn(64).stride(1, 1).convolutionMode(ConvolutionMode.Same)
								.nOut(128).activation(new ActivationReLU()).build(),
						"pool1")
				.addLayer("insert1",
						new ConvolutionLayer.Builder(2, 2).nIn(128).stride(1, 1).convolutionMode(ConvolutionMode.Same)
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

				.addLayer("cnn7", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(128)
						.activation(new ActivationReLU()).convolutionMode(ConvolutionMode.Same).build(), "cnn6")
				// .addVertex("shortcut1", new ElementWiseVertex(Op.Add), "cnn2", "cnn4")

//				.addLayer("cnn4",
//						new ConvolutionLayer.Builder(1, 1).nIn(512).nOut(nBoxes * (5 + numClasses))
//								.weightInit(WeightInit.XAVIER).stride(1, 1).convolutionMode(ConvolutionMode.Same)
//								.weightInit(WeightInit.RELU).activation(Activation.IDENTITY).build(),
//						"insert2")
//				.addLayer("outputs",
//						new Yolo2OutputLayer.Builder().lambbaNoObj(lambdaNoObj).lambdaCoord(lambdaCoord)
//								.boundingBoxPriors(priors).build(),
//						"cnn4")
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
		graphBuilder.addLayer(
				"cnn3Block" + blockNum, new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(filters[2])
						.activation(new ActivationReLU()).convolutionMode(ConvolutionMode.Same).build(),
				"cnn2Block" + blockNum);
		graphBuilder.addVertex("shortcut" + blockNum, new ElementWiseVertex(Op.Add), "cnn1Block" + blockNum,
				"cnn3Block" + blockNum);
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
		graphBuilder.addLayer(
				"cnn3Block" + blockNum, new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(filters[2])
						.activation(new ActivationReLU()).convolutionMode(ConvolutionMode.Same).build(),
				"cnn2Block" + blockNum);
		graphBuilder.addVertex("shortcut" + blockNum, new ElementWiseVertex(Op.Add), "cnn1Block" + blockNum,
				"cnn3Block" + blockNum);
	}

	public static void shortBlock(GraphBuilder graphBuilder, String input, int blockNum, int... filters) {
		graphBuilder.addLayer("cnn1ShortBlock" + blockNum, new ConvolutionLayer.Builder(1, 1).stride(1, 1)
				.nOut(filters[0]).activation(new ActivationReLU()).convolutionMode(ConvolutionMode.Same).build(),
				input);
		graphBuilder.addLayer(
				"cnn2ShortBlock" + blockNum, new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(filters[1])
						.activation(new ActivationReLU()).convolutionMode(ConvolutionMode.Same).build(),
				"cnn1ShortBlock" + blockNum);
		graphBuilder.addVertex("shortcutShortBlock" + blockNum, new ElementWiseVertex(Op.Add),
				"cnn2ShortBlock" + blockNum, input);
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
		// graphBuilder.setOutputs("shortcut" + 4).backprop(true).pretrain(false);

		ComputationGraph neuralNetwork = new ComputationGraph(graphBuilder.build());

		// initialize the network
		neuralNetwork.init();
		System.out.println(neuralNetwork.summary(InputType.convolutional(height, width, numChannels)));

		return neuralNetwork;
	}

	public static void main(String[] args) {
		/**
		 * just to see architecture
		 */
		// Models.CLASModel(216, 112, 3);

		Models.test(216, 112, 3);

	}

}
