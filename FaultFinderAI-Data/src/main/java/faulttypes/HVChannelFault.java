package faulttypes;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

import org.apache.commons.lang3.tuple.Pair;
import org.jlab.groot.data.H1F;
import org.jlab.groot.ui.TCanvas;

import utils.ArrayUtilities;

public class HVChannelFault extends FaultData {

	private List<Pair<Integer, Integer>> hvChannelSegmentation = new ArrayList<Pair<Integer, Integer>>();;
	private Pair<Integer, Integer> randomPair;

	public HVChannelFault() {
		setuphvChannelSegmentation();
		this.xRnd = ThreadLocalRandom.current().nextInt(0, hvChannelSegmentation.size());
		this.faultLocation = this.xRnd;
		this.label = ArrayUtilities.hvChannelFault;
		this.randomPair = getRandomPair();
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
				if (j <= this.randomPair.getRight() - 1 && j >= this.randomPair.getLeft() - 1) {
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
		return this.hvChannelSegmentation.get(this.xRnd);
	}

	private void setuphvChannelSegmentation() {
		hvChannelSegmentation.add(Pair.of(1, 8));
		hvChannelSegmentation.add(Pair.of(9, 16));
		hvChannelSegmentation.add(Pair.of(17, 24));
		hvChannelSegmentation.add(Pair.of(25, 32));
		hvChannelSegmentation.add(Pair.of(33, 48));
		hvChannelSegmentation.add(Pair.of(49, 64));
		hvChannelSegmentation.add(Pair.of(65, 80));
		hvChannelSegmentation.add(Pair.of(81, 112));
	}

	private void makeFaultArray() {
		for (int i = 0; i < this.label.length; i++) {
			if (i == this.xRnd) {
				this.label[i] = 1;
			} else {
				this.label[i] = 0;
			}
		}
		makeReducedLabel();
	}

	/**
	 * The hvChannelSegmentation separation is 8,for List elements 0 - 3
	 * (inclusive) and 16, for List elements 4 - 6 (inclusive) and 32 for List
	 * element 7<br>
	 * Therefore we can try to only use 3 labels and figure out how to get their
	 * positions later
	 */
	private void makeReducedLabel() {
		if (this.faultLocation < 4) {
			this.reducedLabel = IntStream.of(1, 0, 0).toArray();
		} else if (this.faultLocation >= 4 && this.faultLocation < 7) {
			this.reducedLabel = IntStream.of(0, 1, 0).toArray();
		} else {
			this.reducedLabel = IntStream.of(0, 0, 1).toArray();

		}
	}

	public static void main(String[] args) {
		H1F aH1f = new H1F("name", 16, 0, 8);
		for (int i = 0; i < 1000; i++) {
			FaultData faultData = new HVChannelFault();
			aH1f.fill(faultData.getFaultLocation());
			// faultData.plotData();
			// System.out.println(faultData.getFaultLocation() + " " +
			// Arrays.toString(faultData.getReducedLabel()));
		}
		TCanvas canvas = new TCanvas("name", 800, 800);
		canvas.draw(aH1f);

	}

	// @Override
	// public FaultNames getNeighborhood() {
	// if (this.faultLocation < 5) {
	// return FaultNames.CHANNEL_ONE;
	// } else if (this.faultLocation >= 5 && this.faultLocation < 7) {
	// return FaultNames.CHANNEL_TWO;
	// } else {
	// return FaultNames.CHANNEL_THREE;
	// }
	// }
}
