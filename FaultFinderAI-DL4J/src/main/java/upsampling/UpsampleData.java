package upsampling;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.Upsampling1D;
import org.deeplearning4j.nn.conf.layers.Upsampling2D;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import faults.FaultFactory;
import faults.FaultNames;
import lombok.val;
import utils.FaultUtils;

public class UpsampleData {
	private int nExamples = 1;
	private int depth = 20;
	private int nChannelsIn = 1;
	private int inputLength = 28;
	private int size = 4;
	private int outputLength = 28;// inputLength * size;
	private INDArray epsilon = Nd4j.ones(nExamples, depth, outputLength);

	public void testUpsampling1D() throws Exception {

		double[] outArray = new double[] { 1., 1., 2., 2., 3., 3., 4., 4. };
		INDArray containedExpectedOut = Nd4j.create(outArray, new int[] { 1, 1, 8 });
		INDArray containedInput = getContainedData();
		INDArray input = getData();
		Layer layer = getUpsampling1DLayer();
		INDArray containedOutput = layer.activate(containedInput, false, LayerWorkspaceMgr.noWorkspaces());

		INDArray output = layer.activate(input, false, LayerWorkspaceMgr.noWorkspaces());
		FaultUtils.draw(output);

	}

	public void testUpsampling() {

		MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder().list()
				.layer(new ConvolutionLayer.Builder(2, 2).padding(0, 0).stride(2, 2).nIn(1).nOut(3).build()) // (28-2+0)/2+1
																												// = 14
				.layer(new Upsampling2D.Builder().size(3).build()) // 14 * 3 = 42!
				.layer(new OutputLayer.Builder().nOut(3).build()).setInputType(InputType.convolutional(28, 28, 1));

		MultiLayerConfiguration conf = builder.build();

		CnnToFeedForwardPreProcessor proc = (CnnToFeedForwardPreProcessor) conf.getInputPreProcess(2);

	}

	public void testUpsampling1DBackprop() throws Exception {
		INDArray expectedContainedEpsilonInput = Nd4j.create(new double[] { 1., 3., 2., 6., 7., 2., 5., 5. },
				new int[] { 1, 1, 8 });

		INDArray expectedContainedEpsilonResult = Nd4j.create(new double[] { 4., 8., 9., 10. }, new int[] { 1, 1, 4 });

		INDArray input = getContainedData();

		Layer layer = getUpsampling1DLayer();
		layer.activate(input, false, LayerWorkspaceMgr.noWorkspaces());

		Pair<Gradient, INDArray> containedOutput = layer.backpropGradient(expectedContainedEpsilonInput,
				LayerWorkspaceMgr.noWorkspaces());

		INDArray input2 = getData();
		layer.activate(input2, false, LayerWorkspaceMgr.noWorkspaces());
		val depth = input2.size(1);

		epsilon = Nd4j.ones(5, depth, outputLength);

		Pair<Gradient, INDArray> out = layer.backpropGradient(epsilon, LayerWorkspaceMgr.noWorkspaces());

	}

	private Layer getUpsampling1DLayer() {
		NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
				.gradientNormalization(GradientNormalization.RenormalizeL2PerLayer).seed(123)
				.layer(new Upsampling1D.Builder(size).build()).build();
		return conf.getLayer().instantiate(conf, null, 0, null, true);
	}

//	public INDArray getData() throws Exception {
//		DataSetIterator data = new MnistDataSetIterator(5, 5);
//		DataSet mnist = data.next();
//		nExamples = mnist.numExamples();
//		INDArray features = mnist.getFeatureMatrix().reshape(nExamples, nChannelsIn, inputLength, inputLength);
//		return features.slice(0, 3);
//	}
	public INDArray getData() throws Exception {
		FaultFactory factory = new FaultFactory(3, 10, FaultNames.PIN_SMALL, true, true, 1);
		INDArray features = factory.asUnShapedImageMatrix().getImage();
		return features;
	}

	private INDArray getContainedData() {
		INDArray ret = Nd4j.create(new double[] { 1., 2., 3., 4. }, new int[] { 1, 1, 4 });
		FaultUtils.draw(ret);
		return ret;
	}

	public static void main(String[] args) throws Exception {
		UpsampleData upsampleData = new UpsampleData();
		int channels = 3;
		// TCanvas canvas = new TCanvas("aName", 800, 1200);
		// canvas.divide(3, 3);
		// for (int i = 1; i < 10; i++) {
//		FaultFactory factory = new FaultFactory(3, 10, FaultNames.PIN_SMALL, true, true, channels);
//		factory.draw();

		INDArray array = upsampleData.getData();
		upsampleData.testUpsampling1D();
	}

}
