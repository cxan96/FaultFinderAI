package test;

import java.util.concurrent.ThreadLocalRandom;

import org.jlab.groot.data.H2F;
import org.jlab.groot.ui.TCanvas;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.NDArrayUtil;

import strategies.FaultRecordScalerStrategy;
import strategies.MaxStrategy;
import strategies.MinMaxStrategy;
import strategies.SigmoidStrategy;
import strategies.StandardScore;

public class TestScalerStategy {

	int[][] data = new int[6][2];

	public TestScalerStategy() {
		makeData();
	}

	private void makeData() {

		for (int i = 0; i < data[0].length; i++) { // i are the rows (layers)
			for (int j = 0; j < data.length; j++) { // j are the columns (wires)
				int fillNum;
				if ((i > data[0].length - 2) && (j > data.length - 3)) {
					fillNum = ThreadLocalRandom.current().nextInt(50);

				} else {
					fillNum = ThreadLocalRandom.current().nextInt(300, 500);
				}
				data[j][i] = fillNum;
				System.out.println(data[j][i]);

			}
		}
	}

	public void plotData() {
		TCanvas canvas = new TCanvas("Training Data", 800, 1200);
		H2F hData = new H2F("Training Data", 6, 1, 6, 2, 1, 2);
		for (int i = 0; i < data[0].length; i++) { // i are the rows (layers)
			for (int j = 0; j < data.length; j++) { // j are the columns (wires)
				// System.out.println(data[j][i]);
				hData.setBinContent(j, i, data[j][i]);
			}
		}
		canvas.draw(hData);

	}

	public void plotData(FaultRecordScalerStrategy strategy) {

		INDArray features = NDArrayUtil.toNDArray(ArrayUtil.flatten(this.data));
		strategy.normalize(features);
		String canvasTitle = strategy.getClass().getName();
		double[][] data = new double[6][2];

		int rowPlacer = 0;
		for (int i = 0; i < features.length(); i++) {
			double aDub = features.getDouble(i);

			if ((i + 1) % data[0].length == 0) {
				data[rowPlacer][1] = aDub;
				rowPlacer++;
			} else {
				data[rowPlacer][0] = aDub;

			}
		}
		TCanvas canvas = new TCanvas(canvasTitle, 800, 1200);
		H2F hData = new H2F("Training Data", 6, 1, 6, 2, 1, 2);
		for (int i = 0; i < data[0].length; i++) { // i are the rows (layers)
			for (int j = 0; j < data.length; j++) { // j are the columns (wires)
				System.out.println(data[j][i]);
				hData.setBinContent(j, i, data[j][i]);
			}
		}
		canvas.draw(hData);

	}

	public static void main(String[] args) {
		TestScalerStategy test = new TestScalerStategy();
		test.plotData();
		System.out.println("#############  MaxStrategy ###############");
		test.plotData(new MaxStrategy());

		System.out.println("#############  MinMaxStrategy ###############");
		test.plotData(new MinMaxStrategy());

		System.out.println("#############  StandardScore ###############");
		test.plotData(new StandardScore());

		System.out.println("#############  Sigmoid ###############");
		test.plotData(new SigmoidStrategy());

	}
}
