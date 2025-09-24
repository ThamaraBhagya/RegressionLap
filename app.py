import streamlit as st
import pickle
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="centered"
)

# Load the model (cached to avoid reloading on every interaction)
@st.cache_resource
def load_model():
    filename = 'model/predictor.pickle'
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

def prediction(lst):
    model = load_model()
    pred_value = model.predict([lst])
    return pred_value

def create_feature_list(ram, weight, touchscreen, ips, company, typename, opsys, cpu, gpu):
    feature_list = []
    
    # Add numerical features
    feature_list.append(int(ram))
    feature_list.append(float(weight))
    feature_list.append(1 if touchscreen else 0)
    feature_list.append(1 if ips else 0)
    
    # Define category lists
    company_list = ['acer','apple','asus','dell','hp','lenovo','msi','other','toshiba']
    typename_list = ['2in1convertible','gaming','netbook','notebook','ultrabook','workstation']
    opsys_list = ['linux','mac','other','windows']
    cpu_list = ['amd','intelcorei3','intelcorei5','intelcorei7','other']
    gpu_list = ['amd','intel','nvidia']
    
    # Helper function to one-hot encode categories
    def traverse_list(lst, value):
        for item in lst:
            if item == value:
                feature_list.append(1)
            else:
                feature_list.append(0)
    
    # One-hot encode all categorical features
    traverse_list(company_list, company)
    traverse_list(typename_list, typename)
    traverse_list(opsys_list, opsys)
    traverse_list(cpu_list, cpu)
    traverse_list(gpu_list, gpu)
    
    return feature_list

def main():
    # Title and description
    st.title("💻 Laptop Price Predictor")
    st.markdown("Fill in the specifications below to predict the laptop price.")
    
    # Create form for user input
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            ram = st.selectbox(
                "RAM (GB)",
                options=[4, 8, 16, 32, 64],
                index=1  # Default to 8GB
            )
            
            weight = st.number_input(
                "Weight (Kg)",
                min_value=0.5,
                max_value=5.0,
                value=2.0,
                step=0.1
            )
            
            company = st.selectbox(
                "Company",
                options=['', 'acer', 'apple', 'asus', 'dell', 'hp', 'lenovo', 'msi', 'toshiba', 'other'],
                format_func=lambda x: "Select" if x == "" else x.title()
            )
            
            typename = st.selectbox(
                "Type Name",
                options=['', '2in1convertible', 'gaming', 'netbook', 'notebook', 'ultrabook', 'workstation'],
                format_func=lambda x: "Select" if x == "" else {
                    '2in1convertible': '2 in 1 Convertible',
                    'gaming': 'Gaming',
                    'netbook': 'Net Book',
                    'notebook': 'Note Book',
                    'ultrabook': 'Ultra Book',
                    'workstation': 'Workstation'
                }.get(x, x.title())
            )
        
        with col2:
            opsys = st.selectbox(
                "Operating System",
                options=['', 'windows', 'mac', 'linux', 'other'],
                format_func=lambda x: "Select" if x == "" else {
                    'windows': 'Windows',
                    'mac': 'Mac',
                    'linux': 'Linux',
                    'other': 'Other'
                }.get(x, x.title())
            )
            
            cpu = st.selectbox(
                "CPU",
                options=['', 'intelcorei3', 'intelcorei5', 'intelcorei7', 'amd', 'other'],
                format_func=lambda x: "Select" if x == "" else {
                    'intelcorei3': 'Intel Core i3',
                    'intelcorei5': 'Intel Core i5',
                    'intelcorei7': 'Intel Core i7',
                    'amd': 'AMD',
                    'other': 'Other'
                }.get(x, x.title())
            )
            
            gpu = st.selectbox(
                "GPU",
                options=['', 'intel', 'amd', 'nvidia'],
                format_func=lambda x: "Select" if x == "" else {
                    'intel': 'Intel',
                    'amd': 'AMD',
                    'nvidia': 'Nvidia'
                }.get(x, x.title())
            )
            
            st.write("Features:")
            touchscreen = st.checkbox("Touch Screen")
            ips = st.checkbox("IPS Display")
        
        # Submit button
        submitted = st.form_submit_button("Predict Price")
    
    # Handle form submission
    if submitted:
        # Validate required fields
        if not all([company, typename, opsys, cpu, gpu]):
            st.error("Please fill in all the fields.")
            return
        
        try:
            # Create feature list
            feature_list = create_feature_list(
                ram, weight, touchscreen, ips, company, typename, opsys, cpu, gpu
            )
            
            # Make prediction
            pred_value = prediction(feature_list)
            pred_value = np.round(pred_value[0], 2) * 300
            
            # Display result
            st.success(f"💻 Estimated Price: LKR {pred_value:,.2f}")
            
            # Show some additional info
            with st.expander("View Specifications"):
                st.write(f"- **RAM**: {ram} GB")
                st.write(f"- **Weight**: {weight} kg")
                st.write(f"- **Company**: {company.title()}")
                st.write(f"- **Type**: {typename.replace('2in1convertible', '2 in 1 Convertible').title()}")
                st.write(f"- **OS**: {opsys.title()}")
                st.write(f"- **CPU**: {cpu.replace('intelcore', 'Intel Core ').title()}")
                st.write(f"- **GPU**: {gpu.title()}")
                st.write(f"- **Touch Screen**: {'Yes' if touchscreen else 'No'}")
                st.write(f"- **IPS Display**: {'Yes' if ips else 'No'}")
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
    
    # Add some information about the app
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This app predicts laptop prices based on specifications using machine learning.
        
        **Features considered:**
        - RAM capacity
        - Weight
        - Brand/Company
        - Laptop type
        - Operating System
        - CPU model
        - GPU model
        - Touchscreen capability
        - IPS display
        """)
        
        st.header("Instructions")
        st.markdown("""
        1. Fill in all the specifications
        2. Click 'Predict Price'
        3. View the estimated price in LKR
        """)

if __name__ == "__main__":
    main()