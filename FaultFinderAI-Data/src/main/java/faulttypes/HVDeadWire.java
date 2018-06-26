package faulttypes;

import java.util.concurrent.ThreadLocalRandom;

import org.jlab.groot.data.H1F;
import org.jlab.groot.ui.TCanvas;

import utils.ArrayUtilities;

public class HVDeadWire extends FaultData {
	private int[] hvFaultLabel;
	private int faultLocation;

	public HVDeadWire() {
		this.xRnd = ThreadLocalRandom.current().nextInt(1, 113);
		this.yRnd = ThreadLocalRandom.current().nextInt(1, 7);
		this.hvFaultLabel = ArrayUtilities.hvWireFault;
		makeDataSet();
		makeFaultArray();
	}

	@Override
	protected void makeDataSet() {
		for (int i = 0; i < data[0].length; i++) { // i are the rows (layers)
			for (int j = 0; j < data.length; j++) { // j are the columns (wires)
				if (i == (yRnd - 1) && j <= (xRnd - 1) && j >= (xRnd - 1)) {
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

	@Override
	protected int[] getFaultLabel() {
		return hvFaultLabel;
	}

	private void makeFaultArray() {
		this.faultLocation = this.xRnd + ((this.yRnd - 1) * 112) - 1;
		for (int i = 0; i < hvFaultLabel.length; i++) {
			if (i == (faultLocation)) {
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
		for (int i = 0; i < 10; i++) {
			HVDeadWire hvConnectorFault = new HVDeadWire();
			// System.out.println(hvConnectorFault.getFaultLocation() + " iInc
			// ");
			aH1f.fill(hvConnectorFault.getFaultLocation() + 1);
			// hvConnectorFault.plotData();
			int[] fArray = hvConnectorFault.getFaultLabel();
			System.out.println("Fault Location = " + hvConnectorFault.getFaultLocation());
			int count = -1;
			for (int j = 0; j < fArray.length; j++) {
				System.out.print(fArray[j] + " ");
				if (fArray[j] == 1) {
					count = j;
				}
			}
			System.out.println("\n" + count + "   count \n");
		}
		TCanvas canvas = new TCanvas("canvas", 800, 1200);
		canvas.draw(aH1f);

	}
}
