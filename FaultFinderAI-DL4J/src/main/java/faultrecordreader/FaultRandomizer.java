package faultrecordreader;

import java.util.concurrent.ThreadLocalRandom;

import org.jlab.groot.data.H1F;
import org.jlab.groot.ui.TCanvas;

import lombok.Getter;

public class FaultRandomizer {
	@Getter
	private double primaryRnd;
	@Getter
	private int faultNumber;

	public FaultRandomizer() {
		this.primaryRnd = ThreadLocalRandom.current().nextDouble();
		makeFaultNumber();
	}

	private void makeFaultNumber() {
		// 90% of time we want to train on ind.wires
		if (this.primaryRnd > 0.1) {
			if (ThreadLocalRandom.current().nextDouble() > 0.5) {
				this.faultNumber = 4;
			} else {
				this.faultNumber = 5;
			}
		} else {
			this.faultNumber = ThreadLocalRandom.current().nextInt(4);
		}
	}

	public static void main(String[] args) {
		H1F aH1f = new H1F("h", 12, 0, 6);
		for (int i = 0; i < 100000; i++) {
			FaultRandomizer faultRandomizer = new FaultRandomizer();
			aH1f.fill(faultRandomizer.getFaultNumber());
		}
		TCanvas canvas = new TCanvas("CanVas", 800, 800);
		canvas.draw(aH1f);

	}

}
