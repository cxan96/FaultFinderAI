package faulttypes;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

import org.apache.commons.lang3.tuple.Pair;
import org.jlab.groot.data.H1F;
import org.jlab.groot.ui.TCanvas;

import utils.ArrayUtilities;

public class HVPinFault extends FaultData {

	private List<Pair<Integer, Integer>> hvPinSegmentation = new ArrayList<Pair<Integer, Integer>>();

	public HVPinFault() {
		setuphvPinSegmentation();
		this.xRnd = ThreadLocalRandom.current().nextInt(0, hvPinSegmentation.size());
		this.yRnd = ThreadLocalRandom.current().nextInt(0, this.nLayers);
		this.faultLocation = this.xRnd + ((this.yRnd) * 12);
		this.label = ArrayUtilities.hvPinFault;
		/**
		 * reducedLabel is initialized as a IntStream in makeReducedLabel()
		 */
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

	private void makeFaultArray() {
		for (int i = 0; i < label.length; i++) {
			if (i == faultLocation) {
				label[i] = 1;
			} else {
				label[i] = 0;
			}
		}
		makeReducedLabel();
	}

	/**
	 * The hvPinSegmentation separation is 8,for List elements 0 - 9 (inclusive)
	 * and 16, for List elements 9 - 11 (inclusive)<br>
	 * Therefore we can try to only use 2 labels and figure out how to get their
	 * positions later
	 */

	private void makeReducedLabel() {
		if (this.xRnd < this.hvPinSegmentation.size() - 2) {
			this.reducedLabel = IntStream.of(1, 0).toArray();
		} else {
			this.reducedLabel = IntStream.of(0, 1).toArray();
		}
	}

	public static void main(String[] args) {
		H1F aH1f = new H1F("name", 24, 0, 12);
		for (int i = 0; i < 1000; i++) {
			FaultData faultData = new HVPinFault();
			aH1f.fill(faultData.getXRnd());
			// if (faultData.getXRnd()) {
			//
			// }
			System.out.println(faultData.getXRnd() + "  " + faultData.getFaultLocation() + " "
					+ Arrays.toString(faultData.getReducedLabel()));
		}
		TCanvas canvas = new TCanvas("name", 800, 800);
		canvas.draw(aH1f);

	}
}
