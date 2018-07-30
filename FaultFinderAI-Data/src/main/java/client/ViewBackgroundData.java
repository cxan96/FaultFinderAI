package client;

import org.jlab.groot.data.H2F;
import org.jlab.groot.ui.TCanvas;

import utils.FaultUtils;

public class ViewBackgroundData {
	public static void main(String[] args) {
		int[][] data = FaultUtils.dataSector6;
		TCanvas canvas = new TCanvas("Training Data", 800, 1200);
		H2F hData = new H2F("Training Data", 112, 1, 112, 6, 1, 6);
		for (int i = 0; i < data[0].length; i++) { // i are the rows
													// (layers)
			for (int j = 0; j < data.length; j++) { // j are the columns
													// (wires)
				if (i == j % 2) {
					hData.setBinContent(j, i, 0.0);

				} else {
					hData.setBinContent(j, i, data[j][i]);
				}
			}
		}
		canvas.draw(hData);
	}
}
