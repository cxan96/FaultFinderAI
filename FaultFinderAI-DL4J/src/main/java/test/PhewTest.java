package test;

import java.util.HashMap;
import java.util.Map;

public class PhewTest {

	public static String addLayers(Map<String, String> map, int layerNumber) {
		String input = "maxpooling2d_" + (layerNumber - 1);
		System.out.println(input);
		if (!map.containsKey(input)) {
			input = "activation_" + (layerNumber - 1);
			System.out.println("activation part  " + input);
		}
		if (!map.containsKey(input)) {
			input = "concatenate_" + (layerNumber - 1);
			System.out.println("concatenate_ part" + input);

		}
		if (!map.containsKey(input)) {
			input = "input";
			System.out.println("input part " + input);

		}

		return input;
	}

	public static void main(String[] args) {
		Map<String, String> map = new HashMap<>();
		map.put("Yo", "HOHO");
		map.put("A", "Bottle");
		System.out.println("method output   " + addLayers(map, 1));

	}
}
