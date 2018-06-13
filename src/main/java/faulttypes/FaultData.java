package faulttypes;

import org.jlab.groot.data.H2F;
import org.jlab.groot.ui.TCanvas;

public abstract class FaultData {

	protected int xRnd;
	protected int yRnd;

	protected int nLayers = 6;
	protected int rangeMax = 200;
	protected int rangeMin = 100;

	protected int faultRangeMax = 50;
	protected int faultRangeMin = 0;
	// Here I want to see which datasets are needed by DL4J, but for now lets
	// just make it a 2D array.
	protected int[][] data = new int[112][6];

	protected int[][] getData() {
		return data;
	}

	public int getXRand() {
		return xRnd;
	}

	public int getYRand() {
		return yRnd;
	}

	protected abstract void makeDataSet();

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