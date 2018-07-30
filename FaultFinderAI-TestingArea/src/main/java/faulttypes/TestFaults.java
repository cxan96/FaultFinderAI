package faulttypes;

import java.util.ArrayList;
import java.util.List;

public class TestFaults {
	public static void main(String[] args) {
		List<Fault> faultList = new ArrayList<>();
		faultList.add(new HVChannelFault().getInformation());
		faultList.add(new HVFuseFault().getInformation());

		faultList.forEach(x -> {
			System.out.println(x.getFaultName());
			x.getWireInfo().forEach(
					(k, v) -> System.out.println("layer: " + k + " left:" + v.getLeft() + " right: " + v.getRight()));
		});

		Fault newFault = new HVConnectorFault().getInformation();
		System.out.println(newFault.getFaultName() + " ###### newFault");
		newFault.getWireInfo().forEach(
				(k, v) -> System.out.println("layer: " + k + " left:" + v.getLeft() + " right: " + v.getRight()));
		System.out.println("###########################");
		for (Fault fault : faultList) {// get each fault in the list already
			System.out.println(fault.compareFault(newFault));
		}

	}
}
