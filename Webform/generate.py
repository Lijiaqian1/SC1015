while True:
    mainVar = input("thing: ")
    splitedmainVar = mainVar.split(" ")
    if (len(splitedmainVar) == 1):
        shortHand = mainVar.lower()
    else:
        for i in range(len(splitedmainVar)):
            splitedmainVar[i] = splitedmainVar[i][0].upper()
        shortHand ="".join(splitedmainVar)
    

    print(f"<label for=\"{shortHand}\">{mainVar}:</label>")
    print(f"<input type=\"checkbox\" id=\"{shortHand}\" name=\"{shortHand}\">")
