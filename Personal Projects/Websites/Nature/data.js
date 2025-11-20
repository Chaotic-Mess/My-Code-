// Natural landmarks with real coordinates
const locations = [
    // VOLCANOES
    {
        name: "Mount Vesuvius",
        type: "volcano",
        country: "Italy",
        lat: 40.8214,
        lon: 14.4260,
        elevation: "1,281 m (4,203 ft)",
        description: "Famous for the catastrophic eruption in 79 AD that destroyed Pompeii and Herculaneum.",
        wikiUrl: "locations/mount-vesuvius.html",
        imageUrl: "https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Vesuvius_from_Pompeii_%282%29.jpg/500px-Vesuvius_from_Pompeii_%282%29.jpg"
    },
    {
        name: "Mount Fuji",
        type: "volcano",
        country: "Japan",
        lat: 35.3606,
        lon: 138.7274,
        elevation: "3,776 m (12,389 ft)",
        description: "Japan's highest peak and an active stratovolcano, last erupted in 1707.",
        wikiUrl: "locations/mount-fuji.html",
        imageUrl: "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/080103_hakkai_fuji.jpg/500px-080103_hakkai_fuji.jpg"
    },
    {
        name: "Mount Kilimanjaro",
        type: "volcano",
        country: "Tanzania",
        lat: -3.0674,
        lon: 37.3556,
        elevation: "5,895 m (19,341 ft)",
        description: "Africa's highest peak and the world's tallest free-standing mountain.",
        wikiUrl: "locations/mount-kilimanjaro.html",
        imageUrl: "https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Mount_Kilimanjaro.jpg/500px-Mount_Kilimanjaro.jpg"
    },
    {
        name: "Mauna Loa",
        type: "volcano",
        country: "USA (Hawaii)",
        lat: 19.4756,
        lon: -155.6054,
        elevation: "4,169 m (13,678 ft)",
        description: "One of the world's most active volcanoes and the largest volcano on Earth by volume.",
        wikiUrl: "locations/mauna-loa.html",
        imageUrl: "https://upload.wikimedia.org/wikipedia/commons/thumb/0/08/Mauna_Loa_Volcano.jpg/500px-Mauna_Loa_Volcano.jpg"
    },
    {
        name: "Mount Etna",
        type: "volcano",
        country: "Italy (Sicily)",
        lat: 37.7510,
        lon: 14.9934,
        elevation: "3,329 m (10,922 ft)",
        description: "Europe's highest and most active volcano with near-constant activity.",
        wikiUrl: "locations/mount-etna.html",
        imageUrl: "https://upload.wikimedia.org/wikipedia/commons/thumb/7/71/Etna_from_plane.jpg/500px-Etna_from_plane.jpg"
    },
    {
        name: "Krakatoa",
        type: "volcano",
        country: "Indonesia",
        lat: -6.1021,
        lon: 105.4230,
        elevation: "813 m (2,667 ft)",
        description: "Site of one of the deadliest volcanic eruptions in recorded history (1883).",
        wikiUrl: "locations/krakatoa.html",
        imageUrl: "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Krakatoa_eruption_lithograph.jpg/500px-Krakatoa_eruption_lithograph.jpg"
    },
    {
        name: "Mount St. Helens",
        type: "volcano",
        country: "USA (Washington)",
        lat: 46.1914,
        lon: -122.1956,
        elevation: "2,549 m (8,363 ft)",
        description: "Experienced a catastrophic eruption in 1980, reducing its height by 400 meters.",
        wikiUrl: "locations/mount-st-helens.html",
        imageUrl: "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dc/MSH82_st_helens_plume_from_harrys_ridge_05-19-82.jpg/500px-MSH82_st_helens_plume_from_harrys_ridge_05-19-82.jpg"
    },
    {
        name: "Popocatépetl",
        type: "volcano",
        country: "Mexico",
        lat: 19.0225,
        lon: -98.6278,
        elevation: "5,426 m (17,802 ft)",
        description: "Active volcano near Mexico City, one of North America's most dangerous volcanoes.",
        wikiUrl: "locations/popocatepetl.html",
        imageUrl: "https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Popocat%C3%A9petl%2C_M%C3%A9xico%2C_2013-10-02%2C_DD_02.JPG/500px-Popocat%C3%A9petl%2C_M%C3%A9xico%2C_2013-10-02%2C_DD_02.JPG"
    },
    {
        name: "Mount Erebus",
        type: "volcano",
        country: "Antarctica",
        lat: -77.5300,
        lon: 167.1700,
        elevation: "3,794 m (12,448 ft)",
        description: "The world's southernmost active volcano with a persistent lava lake.",
        wikiUrl: "locations/mount-erebus.html",
        imageUrl: "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0a/Mt_Erebus_Aerial_View.jpg/500px-Mt_Erebus_Aerial_View.jpg"
    },
    {
        name: "Eyjafjallajökull",
        type: "volcano",
        country: "Iceland",
        lat: 63.6313,
        lon: -19.6083,
        elevation: "1,651 m (5,417 ft)",
        description: "2010 eruption disrupted air travel across Europe with massive ash clouds.",
        wikiUrl: "locations/eyjafjallajokull.html",
        imageUrl: "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Eyjafjallaj%C3%B6kull_first_crater_20100329.jpg/500px-Eyjafjallaj%C3%B6kull_first_crater_20100329.jpg"
    },
    {
        name: "Cotopaxi",
        type: "volcano",
        country: "Ecuador",
        lat: -0.6850,
        lon: -78.4367,
        elevation: "5,897 m (19,347 ft)",
        description: "One of the world's highest active volcanoes in the Andes.",
        wikiUrl: "locations/cotopaxi.html",
        imageUrl: "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Cotopaxi_2008-06-27_nevado.jpg/500px-Cotopaxi_2008-06-27_nevado.jpg"
    },
    {
        name: "Mount Pinatubo",
        type: "volcano",
        country: "Philippines",
        lat: 15.1300,
        lon: 120.3500,
        elevation: "1,486 m (4,875 ft)",
        description: "1991 eruption was the second-largest volcanic eruption of the 20th century.",
        wikiUrl: "locations/mount-pinatubo.html",
        imageUrl: "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Pinatubo92pinatubo_caldera_crater_lake.jpg/500px-Pinatubo92pinatubo_caldera_crater_lake.jpg"
    },
    
    // MOUNTAINS
    {
        name: "Mount Everest",
        type: "mountain",
        country: "Nepal/China",
        lat: 27.9881,
        lon: 86.9250,
        elevation: "8,849 m (29,032 ft)",
        description: "Earth's highest mountain above sea level, located in the Himalayas.",
        wikiUrl: "locations/mount-everest.html",
        imageUrl: "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Everest_North_Face_toward_Base_Camp_Tibet_Luca_Galuzzi_2006.jpg/500px-Everest_North_Face_toward_Base_Camp_Tibet_Luca_Galuzzi_2006.jpg"
    },
    {
        name: "K2",
        type: "mountain",
        country: "Pakistan/China",
        lat: 35.8825,
        lon: 76.5133,
        elevation: "8,611 m (28,251 ft)",
        description: "Second-highest mountain on Earth, considered more challenging to climb than Everest.",
        wikiUrl: "https://en.wikipedia.org/wiki/K2",
        imageUrl: "https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/K2%2C_Mount_Godwin_Austen%2C_Chogori%2C_Savage_Mountain.jpg/500px-K2%2C_Mount_Godwin_Austen%2C_Chogori%2C_Savage_Mountain.jpg"
    },
    {
        name: "Kangchenjunga",
        type: "mountain",
        country: "Nepal/India",
        lat: 27.7025,
        lon: 88.1475,
        elevation: "8,586 m (28,169 ft)",
        description: "Third-highest mountain in the world, name means 'Five Treasures of Snow'.",
        wikiUrl: "https://en.wikipedia.org/wiki/Kangchenjunga",
        imageUrl: "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Kangchenjunga_from_Goecha_La.jpg/500px-Kangchenjunga_from_Goecha_La.jpg"
    },
    {
        name: "Matterhorn",
        type: "mountain",
        country: "Switzerland/Italy",
        lat: 45.9763,
        lon: 7.6586,
        elevation: "4,478 m (14,692 ft)",
        description: "Iconic pyramid-shaped peak in the Alps, one of the world's most photographed mountains.",
        wikiUrl: "https://en.wikipedia.org/wiki/Matterhorn",
        imageUrl: "https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/Matterhorn_from_Domh%C3%BCtte_-_2.jpg/500px-Matterhorn_from_Domh%C3%BCtte_-_2.jpg"
    },
    {
        name: "Denali",
        type: "mountain",
        country: "USA (Alaska)",
        lat: 63.0692,
        lon: -151.0070,
        elevation: "6,190 m (20,310 ft)",
        description: "Highest mountain peak in North America, formerly known as Mount McKinley.",
        wikiUrl: "https://en.wikipedia.org/wiki/Denali",
        imageUrl: "https://upload.wikimedia.org/wikipedia/commons/thumb/8/84/Denali_Mt_McKinley.jpg/500px-Denali_Mt_McKinley.jpg"
    },
    {
        name: "Mont Blanc",
        type: "mountain",
        country: "France/Italy",
        lat: 45.8326,
        lon: 6.8652,
        elevation: "4,809 m (15,777 ft)",
        description: "Highest mountain in the Alps and Western Europe.",
        wikiUrl: "https://en.wikipedia.org/wiki/Mont_Blanc",
        imageUrl: "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f6/Mont_Blanc_oct_2004.JPG/500px-Mont_Blanc_oct_2004.JPG"
    },
    {
        name: "Aconcagua",
        type: "mountain",
        country: "Argentina",
        lat: -32.6532,
        lon: -70.0109,
        elevation: "6,961 m (22,838 ft)",
        description: "Highest mountain in the Southern and Western Hemispheres.",
        wikiUrl: "https://en.wikipedia.org/wiki/Aconcagua",
        imageUrl: "https://upload.wikimedia.org/wikipedia/commons/thumb/5/52/Aconcagua_Sunset.jpg/500px-Aconcagua_Sunset.jpg"
    },
    {
        name: "Mount Elbrus",
        type: "mountain",
        country: "Russia",
        lat: 43.3499,
        lon: 42.4392,
        elevation: "5,642 m (18,510 ft)",
        description: "Europe's highest peak, located in the Caucasus Mountains.",
        wikiUrl: "https://en.wikipedia.org/wiki/Mount_Elbrus",
        imageUrl: "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/Mount_Elbrus_-_Russia.jpg/500px-Mount_Elbrus_-_Russia.jpg"
    },
    {
        name: "Mount Kosciuszko",
        type: "mountain",
        country: "Australia",
        lat: -36.4560,
        lon: 148.2630,
        elevation: "2,228 m (7,310 ft)",
        description: "Australia's highest mountain, located in the Snowy Mountains.",
        wikiUrl: "https://en.wikipedia.org/wiki/Mount_Kosciuszko",
        imageUrl: "https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/MtKosciuszko.jpg/500px-MtKosciuszko.jpg"
    },
    {
        name: "Mount Vinson",
        type: "mountain",
        country: "Antarctica",
        lat: -78.5250,
        lon: -85.6170,
        elevation: "4,892 m (16,050 ft)",
        description: "Highest peak in Antarctica, part of the Seven Summits challenge.",
        wikiUrl: "https://en.wikipedia.org/wiki/Mount_Vinson",
        imageUrl: "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6b/Mount_Vinson_from_NW_at_Vinson_Plateau_by_Sigurd_Helle.jpg/500px-Mount_Vinson_from_NW_at_Vinson_Plateau_by_Sigurd_Helle.jpg"
    },
    {
        name: "Puncak Jaya",
        type: "mountain",
        country: "Indonesia",
        lat: -4.0833,
        lon: 137.1833,
        elevation: "4,884 m (16,024 ft)",
        description: "Highest peak in Oceania and the highest island peak in the world.",
        wikiUrl: "https://en.wikipedia.org/wiki/Puncak_Jaya",
        imageUrl: "https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Puncak_Jaya_glacier.jpg/500px-Puncak_Jaya_glacier.jpg"
    },
    {
        name: "Mount Logan",
        type: "mountain",
        country: "Canada",
        lat: 60.5672,
        lon: -140.4055,
        elevation: "5,959 m (19,551 ft)",
        description: "Canada's highest peak and the second-highest in North America.",
        wikiUrl: "https://en.wikipedia.org/wiki/Mount_Logan",
        imageUrl: "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Mount_Logan.jpg/500px-Mount_Logan.jpg"
    },
    {
        name: "Lhotse",
        type: "mountain",
        country: "Nepal/China",
        lat: 27.9617,
        lon: 86.9333,
        elevation: "8,516 m (27,940 ft)",
        description: "Fourth-highest mountain in the world, connected to Everest via the South Col.",
        wikiUrl: "https://en.wikipedia.org/wiki/Lhotse",
        imageUrl: "https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/Lhotse-fromChhukung.jpg/500px-Lhotse-fromChhukung.jpg"
    },
    {
        name: "Makalu",
        type: "mountain",
        country: "Nepal/China",
        lat: 27.8892,
        lon: 87.0883,
        elevation: "8,485 m (27,838 ft)",
        description: "Fifth-highest mountain in the world, known for its perfect pyramid structure.",
        wikiUrl: "https://en.wikipedia.org/wiki/Makalu",
        imageUrl: "https://upload.wikimedia.org/wikipedia/commons/thumb/3/35/Makalu.jpg/500px-Makalu.jpg"
    },
    {
        name: "Table Mountain",
        type: "mountain",
        country: "South Africa",
        lat: -33.9628,
        lon: 18.4098,
        elevation: "1,085 m (3,558 ft)",
        description: "Iconic flat-topped mountain overlooking Cape Town, one of the New7Wonders of Nature.",
        wikiUrl: "https://en.wikipedia.org/wiki/Table_Mountain",
        imageUrl: "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f6/Table_mountain_DanieVDM.jpg/500px-Table_mountain_DanieVDM.jpg"
    }
];
