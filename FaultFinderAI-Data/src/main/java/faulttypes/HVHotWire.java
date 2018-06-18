package faulttypes;

import java.util.concurrent.ThreadLocalRandom;

import org.jlab.groot.data.H1F;
import org.jlab.groot.ui.TCanvas;

import arrayUtils.ArrayUtilities;

public class HVHotWire extends FaultData {
	private int[] hvFaultLabel;
	private int faultLocation;

	public HVHotWire() {
		this.xRnd = ThreadLocalRandom.current().nextInt(1, 113);
		this.yRnd = ThreadLocalRandom.current().nextInt(1, 7);
		makeDataSet();
		this.hvFaultLabel = ArrayUtilities.hvWireFault;
		makeDataSet();
		makeFaultArray();
	}

	@Override
	protected void makeDataSet() {
		for (int i = 0; i < data[0].length; i++) { // i are the rows (layers)
			for (int j = 0; j < data.length; j++) { // j are the columns (wires)
				if (i == (yRnd - 1) && j <= (xRnd - 1) && j >= (xRnd - 1)) {
					data[j][i] = makeRandomData(rangeMin, rangeMax);
				} else {
					data[j][i] = makeRandomData(faultRangeMin + 1, faultRangeMax);
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

	@Override
	protected int[] getFaultLabel() {
		return hvFaultLabel;
	}

	private void makeFaultArray() {
		this.faultLocation = this.xRnd + ((this.yRnd - 1) * 112);
		for (int i = 0; i < hvFaultLabel.length; i++) {
			if (i == (faultLocation - 1)) {
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

	public static void main(String[] args) {
		H1F aH1f = new H1F("name", 112 * 12, 0, 112 * 6);
		for (int i = 0; i < 100000; i++) {
			HVHotWire hvConnectorFault = new HVHotWire();
			// System.out.println(hvConnectorFault.getFaultLocation() + " iInc
			// ");
			aH1f.fill(hvConnectorFault.getFaultLocation());
			// hvConnectorFault.plotData();
			// int[] fArray = hvConnectorFault.getFaultLabel();
			// System.out.println("Fault Location = " +
			// hvConnectorFault.getFaultLocation());
			// for (int j = 0; j < fArray.length; j++) {
			// System.out.print(fArray[j] + " ");
			// }
			// System.out.println("");
		}
		TCanvas canvas = new TCanvas("canvas", 800, 1200);
		canvas.draw(aH1f);

	}
}
