package clasDC.factories;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import java.util.zip.DataFormatException;

import clasDC.faults.Fault;
import clasDC.faults.FaultNames;
import lombok.Builder;
import utils.FaultUtils;

public class SingleFaultFactory extends AbstractFaultFactory {

	@Builder
	private SingleFaultFactory(int superLayer, int minFaults, int maxFaults, List<FaultNames> desiredFaults,
			FaultNames desiredFault, double desiredFaultGenRate, boolean randomSuperlayer, boolean randomSmear,
			int nChannels, boolean singleFaultGen) {
		this.superLayer = superLayer;
		this.minFaults = minFaults;
		this.maxFaults = maxFaults;
		this.desiredFault = desiredFault;
		this.desiredFaultGenRate = desiredFaultGenRate;
		this.desiredFaults = desiredFaults;
		this.randomSuperlayer = randomSuperlayer;
		this.randomSmear = randomSmear;
		this.nChannels = nChannels;
		this.singleFaultGen = singleFaultGen;
		initialize();

		loadData();
		generateFaults();
		makeDataSet();
		/**
		 * here I am converting the data set back to x = columns = wires y =
		 * rows = layers
		 */
		convertDataset();

	}

	protected Fault getFault() {
		if (singleFaultGen) {
			return this.getFault(this.desiredFault);
		}

		return this.getFault(desiredFaults.get(ThreadLocalRandom.current().nextInt(0, desiredFaults.size())));
	}

	public int[] getFaultLabel() {
		int[] label = new int[2];
		// lets see if the desired fault is located in the list, if it is, we
		// have the label
		// [1,0]
		// If not the label is
		// [0,1]
		if (faultList.size() == 0) {
			label = IntStream.of(0, 1).toArray();
		} else {
			boolean wantedFound = false;
			for (Fault fault : faultList) {
				if (fault.getSubFaultName().equals(this.desiredFault)) {
					wantedFound = true;
				}
			}
			if (wantedFound) {
				label = IntStream.of(1, 0).toArray();
			} else {
				label = IntStream.of(0, 1).toArray();
			}
		}
		return label;

	}

	public int getSuperLayer() {
		return this.superLayer;
	}

	public List<Fault> getFaultList() {
		return this.faultList;
	}

	public int getLabelInt() {
		getLabelInt(getFaultLabel());
		return this.labelInt;
	}

	public int getLabelInt(int[] labels) {
		for (int i = 0; i < labels.length; i++) {
			if (labels[i] == 1) {
				this.labelInt = i;
				return i;
			}
		}
		this.labelInt = 0;
		return 0;
	}

	public AbstractFaultFactory getNewFactory() {
		AbstractFaultFactory sl = SingleFaultFactory.builder().superLayer(this.superLayer).maxFaults(this.maxFaults)
				.desiredFault(this.desiredFault).desiredFaults(this.desiredFaults).randomSmear(this.randomSmear)
				.nChannels(this.nChannels).singleFaultGen(this.singleFaultGen)
				.desiredFaultGenRate(this.desiredFaultGenRate).build();
		return sl;
	}

	public AbstractFaultFactory getNewFactory(int superLayer) {
		AbstractFaultFactory sl = SingleFaultFactory.builder().superLayer(superLayer).maxFaults(this.maxFaults)
				.desiredFault(this.desiredFault).desiredFaults(this.desiredFaults).randomSmear(this.randomSmear)
				.nChannels(this.nChannels).singleFaultGen(this.singleFaultGen)
				.desiredFaultGenRate(this.desiredFaultGenRate).build();
		return sl;
	}

	public static void main(String[] args) throws DataFormatException {
		FaultNames desiredFault = FaultNames.CHANNEL_THREE;

		AbstractFaultFactory factory = SingleFaultFactory.builder().superLayer(3).minFaults(1).maxFaults(8)
				.desiredFaults(Stream
						.of(FaultNames.CHANNEL_ONE, FaultNames.CHANNEL_TWO, FaultNames.CONNECTOR_E,
								FaultNames.CONNECTOR_TREE, FaultNames.CONNECTOR_THREE, FaultNames.FUSE_A,
								FaultNames.FUSE_B, FaultNames.FUSE_C, FaultNames.PIN_BIG, FaultNames.PIN_SMALL)
						.collect(Collectors.toCollection(ArrayList::new)))
				.randomSmear(true).nChannels(3).singleFaultGen(false).desiredFault(desiredFault)
				.desiredFaultGenRate(0.5).build();
		for (int i = 0; i < 10; i++) {
			FaultUtils.draw(factory.asImageMatrix(), "Fig " + i);
			System.out.println(factory.faultLocationLabels().get(desiredFault) + " desiredFault faultLocationLabels");
			factory = factory.getNewFactory();

		}
	}

}// end
	// of
	// FaultFactory
	// class.
