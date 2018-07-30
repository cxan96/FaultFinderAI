package faulttypes;

import org.jlab.groot.data.H2F;
import org.jlab.groot.ui.TCanvas;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.NDArrayUtil;

import strategies.FaultRecordScalerStrategy;
import utils.ArrayUtilities;
import utils.FaultUtils;

public abstract class FaultData {

	protected int nLayers = ArrayUtilities.nLayers;
	protected int rangeMax = FaultUtils.RANGE_MAX;
	protected int rangeMin = FaultUtils.RANGE_MIN;

	protected int faultRangeMax = FaultUtils.FAULT_RANGE_MAX;
	protected int faultRangeMin = FaultUtils.FAULT_RANGE_MIN;

	protected int xRnd;
	protected int yRnd;
	protected int faultLocation;
	protected int[][] data = new int[112][6];
	protected int[] label;
	protected int[] reducedLabel;

	protected abstract void makeDataSet();

	protected FaultNames faultName;

	public int getXRnd() {
		return xRnd;
	}

	public int getYRnd() {
		return yRnd;
	}

	public int getFaultLocation() {
		return faultLocation;
	}

	public int[][] getData() {
		return data;
	}

	public int[] getLabel() {
		return label;
	}

	public int[] getReducedLabel() {
		return reducedLabel;
	}

	public void plotData() {
		TCanvas canvas = new TCanvas("Training Data", 800, 1200);
		H2F hData = new H2F("Training Data", 112, 1, 112, 6, 1, 6);
		for (int i = 0; i < data[0].length; i++) { // i are the rows (layers)
			for (int j = 0; j < data.length; j++) { // j are the columns (wires)
				hData.setBinContent(j, i, data[j][i]);
			}
		}
		canvas.draw(hData);

	}

	public void plotData(FaultRecordScalerStrategy strategy) {

		INDArray features = NDArrayUtil.toNDArray(ArrayUtil.flatten(this.data));
		strategy.normalize(features);
		double[][] data = new double[112][6];
		int height = data[0].length;

		int rowPlacer = 0;
		int columnPlacer = 0;
		for (int i = 0; i < features.length(); i++) {
			double aDub = features.getDouble(i);
			if ((i + 1) % height == 0) {
				data[columnPlacer][rowPlacer] = aDub;
				rowPlacer = 0;
				columnPlacer++;
			} else {
				data[columnPlacer][rowPlacer] = aDub;
				rowPlacer++;

			}
		}

		TCanvas canvas = new TCanvas("Training Data", 800, 1200);
		H2F hData = new H2F("Training Data", 112, 1, 112, 6, 1, 6);
		for (int i = 0; i < data[0].length; i++) { // i are the rows (layers)
			for (int j = 0; j < data.length; j++) { // j are the columns (wires)
				hData.setBinContent(j, i, data[j][i]);
			}
		}
		canvas.draw(hData);

	}

	// public abstract FaultNames getNeighborhood();
}
