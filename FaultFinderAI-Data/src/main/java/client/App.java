package client;

import java.util.concurrent.ThreadLocalRandom;

import org.jlab.groot.data.H1F;
import org.jlab.groot.data.H2F;
import org.jlab.groot.ui.TCanvas;

import faulttypes.FaultData;
import faulttypes.FaultFactory;

public class App {

	public static void main(String[] args) {
		// FaultData faultData = new HVPinFault();
		//
		// faultData.plotData();
		TCanvas canvas = new TCanvas("Training Data", 800, 1200);
		H2F hData = new H2F("Training Data", 8, 1, 9, 1, 1, 6);
		H1F hFaults = new H1F("NFaults", 6, 1, 7);
		// for (int i = 0; i < 1; i++) {
		FaultFactory factory = new FaultFactory();
		// FaultData fault = new HVPinFault();
		// FaultData fault = new HVChannelFault();
		int rndFault = ThreadLocalRandom.current().nextInt(0, 6);
		FaultData fault = factory.getFault(0);
		int[] faultLabel = factory.getLabel();
		for (int i = 0; i < faultLabel.length; i++) {
			System.out.print(faultLabel[i] + "  ");
		}
		System.out.println(" \n" + fault.getClass().toString() + "  " + fault.getFaultLocation());
		hFaults.fill(rndFault + 1);
		// fault.plotData();
		// new HVChannelFault().plotData();
		// System.out.println((fault.getXRand() + 1) + " " +
		// (fault.getYRand() + 1));
		// hData.fill(fault.getXRand() + 1, fault.getYRand() + 1);
		// }
		canvas.draw(hFaults);

	}

}
