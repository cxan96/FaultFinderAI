package faulttypes;

import org.jlab.groot.data.H2F;
import org.jlab.groot.ui.TCanvas;

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

}
