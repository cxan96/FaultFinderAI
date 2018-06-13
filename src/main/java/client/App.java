package client;

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
		for (int i = 0; i < 150; i++) {
			FaultFactory factory = new FaultFactory();
			// FaultData fault = new HVPinFault();
			// FaultData fault = new HVChannelFault();
			FaultData fault = factory.getFault(2);

			// new HVChannelFault().plotData();
			System.out.println((fault.getXRand() + 1) + "  " + (fault.getYRand() + 1));
			hData.fill(fault.getXRand() + 1, fault.getYRand() + 1);
		}
		canvas.draw(hData);

	}

}
