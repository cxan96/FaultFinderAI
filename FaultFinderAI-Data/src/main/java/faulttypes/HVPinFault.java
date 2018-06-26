package faulttypes;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.lang3.tuple.Pair;

import utils.ArrayUtilities;

public class HVPinFault extends FaultData {

	private List<Pair<Integer, Integer>> hvPinSegmentation = new ArrayList<Pair<Integer, Integer>>();
	private int[] hvFaultLabel;
	private int faultLocation;

	public HVPinFault() {
		setuphvPinSegmentation();
		this.xRnd = ThreadLocalRandom.current().nextInt(0, hvPinSegmentation.size());
		this.yRnd = ThreadLocalRandom.current().nextInt(0, this.nLayers);
		this.hvFaultLabel = ArrayUtilities.hvPinFault;

		makeDataSet();
		makeFaultArray();
	}

	@Override
	public void makeDataSet() {
		for (int i = 0; i < data[0].length; i++) { // i are the rows (layers)
			for (int j = 0; j < data.length; j++) { // j are the columns (wires)
				if (i == yRnd && j <= getRandomPair().getRight() - 1 && j >= getRandomPair().getLeft() - 1) {
					data[j][i] = makeRandomData(faultRangeMin, faultRangeMax);
				} else {
					data[j][i] = makeRandomData(rangeMin, rangeMax);
				}
				// System.out.print(data[j][i] + "\t");
				// if (j == data.length - 1) {
				// System.out.println("");
				// }
			}
		}
	}

	private int makeRandomData(int rangeMin, int rangeMax) {
		return ThreadLocalRandom.current().nextInt(rangeMin, rangeMax + 1);
	}

	private Pair<Integer, Integer> getRandomPair() {
		return this.hvPinSegmentation.get(this.xRnd);
	}

	private void setuphvPinSegmentation() {
		hvPinSegmentation.add(Pair.of(1, 8));
		hvPinSegmentation.add(Pair.of(9, 16));
		hvPinSegmentation.add(Pair.of(17, 24));
		hvPinSegmentation.add(Pair.of(25, 32));
		hvPinSegmentation.add(Pair.of(33, 40));
		hvPinSegmentation.add(Pair.of(41, 48));
		hvPinSegmentation.add(Pair.of(49, 56));
		hvPinSegmentation.add(Pair.of(57, 64));
		hvPinSegmentation.add(Pair.of(65, 72));
		hvPinSegmentation.add(Pair.of(73, 80));
		hvPinSegmentation.add(Pair.of(81, 96));
		hvPinSegmentation.add(Pair.of(97, 112));
	}

	@Override
	protected int[] getFaultLabel() {
		return hvFaultLabel;
	}

	private void makeFaultArray() {
		this.faultLocation = this.xRnd + ((this.yRnd) * 12);
		for (int i = 0; i < hvFaultLabel.length; i++) {
			if (i == faultLocation) {
				hvFaultLabel[i] = 1;
			} else {
				hvFaultLabel[i] = 0;
			}
		}
	}

	@Override
	public int getFaultLocation() {
		return this.faultLocation;
	}

	// public static void main(String[] args) {
	// H1F aH1f = new H1F("name", 72 * 2, 0, 72);
	// for (int i = 0; i < 100000; i++) {
	// HVPinFault hvConnectorFault = new HVPinFault();
	// // System.out.println(hvConnectorFault.getFaultLocation() + " iInc
	// // ");
	// aH1f.fill(hvConnectorFault.getFaultLocation());
	// // hvConnectorFault.plotData();
	// // int[] fArray = hvConnectorFault.getFaultLabel();
	// // System.out.println("Fault Location = " +
	// // hvConnectorFault.getFaultLocation());
	// // for (int j = 0; j < fArray.length; j++) {
	// // System.out.print(fArray[j] + " ");
	// // }
	// // System.out.println("");
	// }
	// TCanvas canvas = new TCanvas("canvas", 800, 1200);
	// canvas.draw(aH1f);
	//
	// }

}
