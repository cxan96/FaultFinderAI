package clasDC.faults;

import java.util.List;

import lombok.Builder;

public class UNetFaultFactory extends AbstractFaultFactory {

	@Builder
	public UNetFaultFactory(int superLayer, int maxFaults, List<FaultNames> desiredFault, boolean randomSuperlayer,
			boolean randomSmear, int nChannels, boolean singleFaultGen) {
		this.superLayer = superLayer;
		this.maxFaults = maxFaults;
		this.desiredFaults = desiredFault;
		this.randomSuperlayer = randomSuperlayer;
		this.randomSmear = randomSmear;
		this.nChannels = nChannels;
		this.singleFaultGen = singleFaultGen;
		initialize();

		loadData();
		generateFaults();
		makeDataSet();
		/**
		 * here I am converting the data set back to x = columns = wires y = rows =
		 * layers
		 */
		convertDataset();

	}

	@Override
	protected Fault getFault() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	protected int[] getFaultLabel() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public AbstractFaultFactory getNewFactory() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public AbstractFaultFactory getNewFactory(int superLayer) {
		// TODO Auto-generated method stub
		return null;
	}

}
