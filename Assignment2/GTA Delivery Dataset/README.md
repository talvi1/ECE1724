# GTA Delivery Dataset

The dataset contains 175 delivery records for addresses in the Greater Toronto Area (GTA). 
The entries are based on "real-world" delivery information, similar to what a typical small-size Delivery Service Provider (DSP) might face in a given operating day.

### order_id
A UUID4 identifier for each delivery transaction, as a method of referencing different deliveries. 
This is required as there may be more than one delivery going from the same origin to the same destination in a given day (multiple packages, or multiple orders from the same fulfillment centre).

### collection_lat and collection_lng
Latitude and Longitude for the pickup location. This is generally some sort of warehouse or fulfillment centre where delivery drivers will pickup their packages and return to after their route is completed.
In some cases, this may be a third-party location (for example, picking up packages directly from the shipper).

### dropoff_lat and dropoff_lng
Latitude and Longitude for the target address. These locations can be residential, commercial, or institutional in nature, and thus vary in terms of delivery complexity.
Does the driver simply drop off the package and leave, or do they need to complete some sort of package sign in process, call the customer, etc? These can all affect the time it takes to complete a delivery.

### due_date
The promised "deliver by" time. While this time is a promise made by the carrier to the shipper, it is not a hard limit. See **hard_limit** for absolute "deliver before" time.

### weight
The approximated weight of the package, in kg. This can be used when optimizing for vehicles with limited capacity. 
While package dimensions are not provided, you can reverse engineer a somewhat reasonable package volume using the Volumetric Weight equation:

L x W x H = Weight x 5

### max_load_time
The maximum amount of time a package can sit in the delivery vehicle, due to some sort of physical constraint. 
For example, non-refrigerated groceries or meal kits packed in insulated boxes normally have a maximum lifespan of 16 hours, half of which is reserved to account for the package sitting unattended on the customer's front porch or building lobby. In this case, the package can only sit in the delivery vehicle for a maximum of 8 hours.
While this is an important constraint for the delivery carrier, it is not always enforced very strictly, and some minor deviations are allowed (15 mins delay, etc.)

### hard_limit
This is a hard limit for delivery deadline. This can occur for a variety of reasons, such as a customer noting that they will not accept packages beyond a certain time, or a requirement from the shipper as part of their agreement with the carrier. For commercial locations, this may be when the business closes for the day.



## Use for non-delivery applications
The same dataset can be used for optimization problems involving emergency services dispatching or ridesharing. For these applications, the fields related to package delivery can be ignored.



