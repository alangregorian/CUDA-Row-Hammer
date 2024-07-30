# Pointer Chase Test
This is a CUDA implementation of fine-grained pointer chasing that accepts the following arguments: size, stride, iterations, and random. It initializes an array using the size and spaces the elements apart according to the stride. The array is traversed starting at index 0, where the value of the current element is used to re-index into the array. The values of these elements are initialized to the index of the next in-order element of the array, unless the random argument is provided.

## Build
```
make all
```

## Usage
```
Usage: ./pointer_chase [options]
Options:
  -h (--help)       	Show help message
  -i (--iterations) 	Set number of iterations
  -n (--size)       	Set array size
  -r (--random)     	Enable random indices
  -s (--stride)     	Set stride value
```

## Example Run
```
./pointer_chase -n $((128*1024*1024)) -s $((4*1024)) -i 64 -r
```

## Example Output
|Size     |Index    |Cycles|
|---------|---------|------|
|134217728|0        |1126  |
|134217728|8851456  |1156  |
|134217728|131698688|813   |
|134217728|36757504 |1172  |
|134217728|28909568 |1167  |
|134217728|58998784 |1160  |
|134217728|94347264 |877   |
|134217728|68231168 |1126  |
|134217728|133357568|311   |
|134217728|86003712 |844   |
|134217728|117518336|859   |
|134217728|30371840 |641   |
|134217728|44969984 |1156  |
|134217728|58359808 |1183  |
|134217728|132947968|636   |
|134217728|46931968 |601   |
|134217728|118353920|616   |
|134217728|724992   |615   |
|134217728|132820992|610   |
|134217728|91680768 |618   |
|134217728|76398592 |1118  |
|134217728|120680448|654   |
|134217728|626688   |629   |
|134217728|118538240|616   |
|134217728|66248704 |632   |
|134217728|87285760 |668   |
|134217728|35340288 |606   |
|134217728|111628288|1228  |
|134217728|38719488 |635   |
|134217728|13103104 |643   |
|134217728|65310720 |839   |
|134217728|132288512|644   |
|134217728|6164480  |589   |
|134217728|43433984 |617   |
|134217728|130449408|653   |
|134217728|126726144|641   |
|134217728|9555968  |647   |
|134217728|83681280 |645   |
|134217728|91627520 |652   |
|134217728|80949248 |601   |
|134217728|97894400 |675   |
|134217728|128983040|809   |
|134217728|132325376|624   |
|134217728|44359680 |634   |
|134217728|115400704|626   |
|134217728|105947136|1131  |
|134217728|49418240 |666   |
|134217728|51605504 |681   |
|134217728|113106944|657   |
|134217728|50561024 |622   |
|134217728|10559488 |615   |
|134217728|2019328  |670   |
|134217728|24109056 |1348  |
|134217728|33849344 |665   |
|134217728|48885760 |625   |
|134217728|50290688 |643   |
|134217728|131076096|634   |
|134217728|8122368  |635   |
|134217728|15884288 |615   |
|134217728|121487360|678   |
|134217728|88858624 |611   |
|134217728|133152768|677   |
|134217728|112513024|648   |
|134217728|119025664|655   |